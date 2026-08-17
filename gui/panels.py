"""대시보드 데이터 로더 — 순수 stdlib(pandas/streamlit 미의존, 테스트 가능).

일일 루프가 남긴 관측 스냅샷(`data/state/observations/`)과 결정 로그
(`data/logs/`)를 읽어 UI 가 바로 쓰는 평범한 dict/list 로 정규화한다.
브로커 API 를 치지 않고, 파일만 읽는다(결정론·감사 가능).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from eval.meta import load_arm_history
from harness.jsonlog import iter_events


def list_observation_days(obs_dir: Path, market: str) -> list[str]:
    """해당 시장의 관측 스냅샷 날짜(YYYY-MM-DD)를 최신순으로 반환."""

    market_dir = obs_dir / market
    if not market_dir.exists():
        return []
    return sorted((p.stem for p in market_dir.glob("*.json")), reverse=True)


def load_observation(obs_dir: Path, market: str, day: str) -> dict | None:
    """{obs_dir}/{market}/{day}.json 스냅샷을 읽는다. 없으면 None."""

    path = obs_dir / market / f"{day}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_recent_decisions(log_dir: Path, market: str, limit: int = 30) -> list[dict]:
    """daily_step 로그 → asof_day 오름차순 정규화 결정 행.

    로그 파일명은 실행일이지만 여기서는 레코드의 asof_day 로 키를 잡는다
    (관측 스냅샷과 같은 축). 같은 asof 재실행은 최신 레코드로 덮어쓴다.
    """

    rows: dict[str, dict] = {}
    for rec in iter_events(log_dir, market, "daily_step"):
        day = str(rec.get("asof_day", ""))[:10]
        if not day:
            continue
        d = rec.get("decision") or {}
        rows[day] = {
            "day": day,
            "policy": rec.get("policy"),
            "weights": rec.get("weights") or {},
            "accepted": rec.get("accepted"),
            "features": d.get("features") or {},
            "rationale": d.get("rationale"),
            "debate": d.get("debate"),
            "risk_violations": d.get("risk_violations") or [],
            "weights_pre_risk": d.get("weights_pre_risk"),
            "mdd": d.get("mdd"),
            "cited_signal_ids": d.get("cited_signal_ids") or [],
            "cited_memory_ids": d.get("cited_memory_ids") or [],
            "scenario_expected": d.get("scenario_expected"),
            "scenario_invalidation": d.get("scenario_invalidation"),
            "budget": d.get("budget"),  # None = 그날 예산 조회 실패 또는 순자산 0
        }
    return [rows[k] for k in sorted(rows)][-limit:]


def latest_budget(decisions: list[dict]) -> tuple[str, dict] | None:
    """가장 최근에 **실제로 기록된** 예산 스냅샷을 (기록일, 스냅샷) 으로. 없으면 None.

    순자산이 0 이거나(미입금) 계좌 조회가 실패한 날은 예산이 null 로 남는다. 최신 행만
    집으면 그런 날 하루에 근거가 통째로 사라지므로 거슬러 찾되, **언제 것인지 함께**
    돌려준다 — 날짜 없이 보여주면 오래된 잔고를 현재 잔고로 읽는다.
    """

    for row in reversed(decisions):
        budget = row.get("budget")
        if budget:
            return row["day"], budget
    return None


# 1주 비중이 종목당 상한의 이 배수보다 크면 살 수 있는 수량이 사실상 0·1·2주뿐이라
# 비중을 조절할 수 없다. 유니버스를 고를 때 쓴 등급 기준을 그대로 표시에도 쓴다.
_COARSE_STEP_RATIO = 3.0


def universe_rows(
    meta: dict[str, dict],
    equity_cap: float,
    asset_caps: dict[str, float],
    budget: dict | None,
    closes: dict[str, float],
) -> list[dict]:
    """관측·집행 유니버스를 종목별 한 행으로. 집행 가능 종목이 먼저 온다.

    meta 는 트레이더 프롬프트에 실제로 주입되는 종목 메타(`adapters.universe.universe_meta`)
    를 그대로 받는다 — 화면과 프롬프트가 같은 값을 보게 하기 위해서다.

    판정은 예산 스냅샷의 1주 비중(min_step_weight)과 종목당 상한을 비교한 것이다.
    1주 비중이 상한보다 크면 그 종목에는 어떤 비중도 표현할 수 없어 집행 유니버스에서
    빠진다 — 집행이 좁은 이유가 전략이 아니라 계좌 크기임을 이 열이 보여준다.
    """

    steps = (budget or {}).get("min_step_weight") or {}
    # 분수 거래 계좌는 거래단위가 없어 종목별 1주 비중을 **애초에 기록하지 않는다**.
    # 그 빈칸을 "기록 없음"으로 읽으면 정상 동작하는 계좌가 고장난 것처럼 보이므로,
    # 종목별 유무가 아니라 계좌 단위 사실인 lot 으로 가른다.
    fractional = budget is not None and not (budget.get("lot") or 0)
    rows: list[dict] = []
    for symbol, block in meta.items():
        cap = asset_caps.get(symbol, equity_cap)
        step = steps.get(symbol)
        close = closes.get(symbol)
        # 1주가 / 순자산 ≤ 상한 을 순자산에 대해 푼 값. 담을 수 없는(또는 알 수 없는)
        # 종목에만 채운다 — 이미 담기는 종목에는 답이 아니라 잡음이다.
        need_equity = None
        if close and cap > 0 and not fractional and (step is None or step > cap):
            need_equity = round(close / cap, 2)
        if fractional:
            verdict = "분수 거래 (단위 제약 없음)"
        elif budget is None:
            verdict = "예산 기록 없음"
        elif step is None:
            verdict = "1주 비중 미기록"
        elif step > cap:
            verdict = "편입 불가 (1주가 상한 초과)"
        elif step > cap / _COARSE_STEP_RATIO:
            verdict = "거칠게만 (0·1·2주)"
        else:
            verdict = "조절 가능"
        rows.append({
            "symbol": symbol,
            "asset_class": block.get("asset_class"),
            "index": block.get("index"),  # ETF 만 채워진다
            "tradable": bool(block.get("tradable")),
            "cap": cap,
            "min_step_weight": step,
            "last_close": close,
            "min_equity_for_entry": need_equity,
            "verdict": verdict,
        })
    rows.sort(key=lambda r: not r["tradable"])  # 집행 가능 먼저 (그 안에서는 유니버스 순서)
    return rows


def exposure_turnover(decisions: list[dict]) -> list[dict]:
    """정규화된 결정 행 → 일별 현금비중·투자비중·turnover(직전 대비 L1/2). 첫날 turnover=None."""
    rows: list[dict] = []
    prev: dict[str, float] | None = None
    for d in decisions:
        w = d.get("weights") or {}
        cash = float(w.get("CASH", 0.0))
        turnover = None
        if prev is not None:
            turnover = 0.5 * sum(
                abs(float(w.get(k, 0.0)) - float(prev.get(k, 0.0))) for k in set(w) | set(prev)
            )
        rows.append({"day": d["day"], "cash": cash, "invested": 1.0 - cash, "turnover": turnover})
        prev = w
    return rows


def scenario_outcomes(decisions: list[dict], equity: list[dict]) -> list[dict]:
    """결정의 시나리오(예상·무효화 조건)를 익일 실현 수익률과 나란히.

    무효화 조건은 자연어라 자동 판정하지 않는다 — 익일 arm 수익률을 함께 보여
    사람이 무효화 여부를 읽게 한다(정직한 표시, 자동 verdict 없음).
    """
    eq_by_day = {p["day"]: p["equity"] for p in equity}
    days = sorted(eq_by_day)
    idx = {d: i for i, d in enumerate(days)}
    rows: list[dict] = []
    for d in decisions:
        exp, inv = d.get("scenario_expected"), d.get("scenario_invalidation")
        if not (exp or inv):
            continue
        day = d["day"]
        nxt = None
        i = idx.get(day)
        if i is not None and i + 1 < len(days):
            e0, e1 = eq_by_day[days[i]], eq_by_day[days[i + 1]]
            if e0:
                nxt = e1 / e0 - 1.0
        rows.append({"day": day, "expected": exp, "invalidation": inv, "next_day_return": nxt})
    return rows


def decision_for_day(rows: list[dict], day: str) -> dict | None:
    """정규화된 결정 행 목록에서 특정 날짜의 결정을 찾는다."""

    return next((r for r in rows if r["day"] == day), None)


def veto_rows(rows: list[dict]) -> list[dict]:
    """risk 위반이 있었던 날만 추려 타임라인 표로."""

    return [
        {"day": r["day"], "violations": "; ".join(r["risk_violations"]), "mdd": r.get("mdd")}
        for r in rows
        if r["risk_violations"]
    ]


# ── 국면(regime) · 안전/헬스 (읽기 전용) ──


def load_regime(state_dir: Path) -> dict:
    """regime_latest.json — {market: {state, drawdown, asof}}. 없으면 {}. shadow 신호."""
    path = state_dir / "regime_latest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def kill_switch_active(state_dir: Path) -> bool:
    """실주문 정지 파일(KILL_SWITCH) 존재 여부 — `touch data/state/KILL_SWITCH` 로 발동."""
    return (state_dir / "KILL_SWITCH").exists()


def market_health(log_dir: Path, market: str, today: date) -> dict:
    """시장별 운영 헬스 — 마지막 결정 신선도 + 리스크 엔진 MDD + 최근 위반.

    staleness 는 결정 로그의 asof_day 기준(마지막 결정이 며칠 전인가). mdd 는 리스크
    엔진이 서킷브레이커 판정에 쓰는 값(가상 arm 낙폭과는 다른 기준). 파일만 읽는다.
    """
    decisions = read_recent_decisions(log_dir, market)
    last = decisions[-1] if decisions else None
    last_day = last["day"] if last else None
    days_stale: int | None = None
    if last_day:
        try:
            days_stale = (today - date.fromisoformat(last_day)).days
        except ValueError:
            days_stale = None
    return {
        "market": market,
        "last_day": last_day,
        "days_stale": days_stale,
        "mdd": last.get("mdd") if last else None,
        "violation_days": [d["day"] for d in decisions if d["risk_violations"]][-3:],
    }


def latest_meta_event(log_dir: Path) -> dict | None:
    """META 네임스페이스 최신 meta_shadow 이벤트(제안 가중치·틸트 근거). 없으면 None."""
    events = list(iter_events(log_dir, "META", "meta_shadow"))
    return events[-1] if events else None


def treasury_dryrun_report(log_dir: Path) -> dict | None:
    """TREASURY 최신 dry-run plan + 같은 실행의 이체 의도들. 없으면 None."""
    plans = list(iter_events(log_dir, "TREASURY", "treasury_dryrun_plan"))
    if not plans:
        return None
    last = plans[-1]
    day = str(last.get("ts", ""))[:10]
    intents = [
        i for i in iter_events(log_dir, "TREASURY", "treasury_dryrun_intent")
        if str(i.get("ts", ""))[:10] == day
    ]
    return {"plan": last, "intents": intents}


def load_latest_requests(requests_dir: Path) -> dict | None:
    """가장 최근 달의 능력 갭 요구 파일({YYYY-MM}.json)을 읽는다. 없으면 None."""

    if not requests_dir.exists():
        return None
    files = sorted(requests_dir.glob("*.json"), reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text(encoding="utf-8"))


# ── 제안서 (월간 자동 / 세션발 초안) ──

_MONTHLY = re.compile(r"\d{4}-\d{2}\.md")


def monthly_proposals(proposal_dir: Path) -> list[str]:
    """월간 self-improve 제안서 파일명({YYYY-MM}.md) — 최신순."""

    if not proposal_dir.exists():
        return []
    return sorted((p.name for p in proposal_dir.glob("*.md") if _MONTHLY.fullmatch(p.name)),
                  reverse=True)


def _frontmatter(text: str) -> dict[str, str]:
    """`---` 사이의 `key: value` 만 뽑는다. 없으면 빈 dict."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def session_proposals(proposal_dir: Path) -> list[dict]:
    """토론 세션에서 나온 플레이북 초안 — 최신순.

    월간 자동 제안서와 **섞어 보여주지 않는다**. 둘은 출처가 다르다 — 하나는 승격
    통계·성과에서 나온 것이고 하나는 사람과의 대화에서 나온 것이라, 한 목록에 두면
    나중에 어느 쪽이 무엇이었는지 가릴 수 없다.
    """

    if not proposal_dir.exists():
        return []
    rows = []
    for path in sorted(proposal_dir.glob("session-*.md"), reverse=True):
        meta = _frontmatter(path.read_text(encoding="utf-8"))
        cited = meta.get("cited_ids", "").strip("[]")
        rows.append({
            "name": path.name,
            "created": meta.get("created", ""),
            "market": meta.get("market", ""),
            "session_id": meta.get("session_id", ""),
            "target": meta.get("target", ""),
            # 근거 없이 나온 초안을 목록에서 바로 가릴 수 있어야 한다
            "grounding": meta.get("grounding", "none"),
            "n_cited": len([c for c in cited.split(",") if c.strip()]),
            "applies": meta.get("applies") == "true",
            "applies_reason": meta.get("applies_reason", ""),
            "applied": meta.get("applied") == "true",
        })
    return rows


def session_draft_diff(path: Path) -> str:
    """초안 파일에서 diff 본문만 꺼낸다 — frontmatter 를 markdown 으로 렌더하면
    `---` 가 제목 문법으로 읽혀 헤더가 거대한 제목처럼 보인다."""

    body = path.read_text(encoding="utf-8")
    _, fence, rest = body.partition("```diff\n")
    return rest.rpartition("```")[0].rstrip() if fence else ""


# ── 자본 배분 (파이 데이터) ──


def load_market_allocation(virtual_dir: Path, meta_ledger: Path, arm: str = "llm") -> dict:
    """마켓별 자본 배분 — 현재(가상 equity 비중) vs 목표(meta_shadow 최신 제안).

    현재 비중은 arm equity 를 시장별로 정규화(전 시장 동일 nominal base 라 공통 단위).
    목표는 meta_shadow 원장의 최신 제안 weights. 각각 없으면 빈 dict.
    """
    suffix = f"_{arm}.json"
    equities: dict[str, float] = {}
    if virtual_dir.exists():
        for path in sorted(virtual_dir.glob(f"*{suffix}")):
            market = path.name[: -len(suffix)]
            history = load_arm_history(virtual_dir, market, arm)
            eq = history[-1]["equity"] if history else None
            if eq and eq > 0:
                equities[market] = eq
    total = sum(equities.values())
    current = {m: eq / total for m, eq in equities.items()} if total > 0 else {}

    target: dict[str, float] = {}
    if meta_ledger.exists():
        history = json.loads(meta_ledger.read_text(encoding="utf-8")).get("history") or []
        if history:
            target = history[-1].get("weights") or {}
    return {"current": current, "target": target, "arm": arm}


def load_intramarket_weights(state_dir: Path, market: str) -> dict[str, float]:
    """마켓 내 목표 배분 벡터(CASH 포함) — risk_{market}.json 의 prev_weights. 없으면 {}."""
    path = state_dir / f"risk_{market}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("prev_weights") or {}


# ── 스케줄 잡 상태 (launchd 로그 기반) ──

# briefing= : paper_step, status=ok : 범용, cycle_done : alpha_lab 사이클 완료.
# 시장별 상태를 파싱해 롤업한다 — 일부 시장만 실패하면 error 가 아니라 partial(한 시장의
# 네트워크 블립이 같은 잡의 다른 시장을 실패로 오염시키는 것 방지). 아래 범용 마커는
# 시장 라인이 없는 잡(proposal·requests)의 폴백.
_OK_MARKERS = ("briefing=", "status=ok", "cycle_done")
_FAIL_MARKERS = ("status=fail", "status=error")

# 시장별 상태 토큰: "market=US status=ok" 와 "[CRYPTO] status=error" 를 모두 매칭.
_MKT_STATUS = re.compile(r"(?:market=|\[)([A-Za-z_]+)\]?\s+status=(\w+)")
_MKT_CYCLE_OK = re.compile(r"\[([A-Za-z_]+)\]\s+cycle_done")  # alpha_lab 시장 완료
# observed·triggered 등 중간 상태는 무시(최신 '종결' 만 반영).
# no_trigger(워처 무발동 틱)·closed(장 마감 — 워처 게이팅 스킵 / 브로커의 비영업일 거부)
# =정상, rejected(장은 열렸는데 주문이 거부됨)=실패.
# degraded=브로커 장애로 실주문만 스킵(결정·가상 arm 은 정상) — 주의가 필요하니 실패로 센다.
_TERMINAL = {"ok", "error", "fail", "no_trigger", "closed", "rejected", "degraded"}


def _market_statuses(out_tail: list[str]) -> dict[str, str]:
    """out 로그 tail 에서 시장별 최신 종결 상태(ok|error)를 뽑는다(마지막 등장 우선).

    status= 없는 유사 시장 라인(MACRO·META·virtual·regime)과 중간 상태(observed)는 무시.
    """
    latest: dict[str, str] = {}
    for line in out_tail:
        m = _MKT_STATUS.search(line)
        if m and m.group(2) in _TERMINAL:
            latest[m.group(1)] = (
                "error" if m.group(2) in ("error", "fail", "rejected", "degraded") else "ok"
            )
            continue
        c = _MKT_CYCLE_OK.search(line)
        if c:
            latest[c.group(1)] = "ok"
    return latest


def running_jobs(state_dir: Path, now: datetime | None = None) -> list[dict]:
    """지금 돌고 있는 잡 목록 — 락 파일의 실행 표식에서 읽는다(락은 건드리지 않는다).

    out 로그의 `status=ok` 는 **그 시장의 주문 왕복이 끝났다**는 뜻일 뿐, 런 전체의 종료가
    아니다. 뒤이어 메모리 파이프라인·주간 회고·shadow 채널이 더 돌기 때문에, 로그 tail 만
    보면 아직 진행 중인데도 완료로 읽힌다 — 그 상태에서 기기를 재우면 남은 단계가 통째로
    유실된다. 그래서 완료 추정과 별개로 '살아 있는 프로세스'를 직접 확인한다.
    """
    from harness.lock import read_run_marker

    now = now or datetime.now(timezone.utc)
    out: list[dict] = []
    if not state_dir.exists():
        return out
    for path in sorted(state_dir.glob("*.lock")):
        marker = read_run_marker(path)
        if not marker:
            continue
        elapsed = None
        try:
            started = datetime.fromisoformat(marker["started_at"])
            elapsed = int((now - started).total_seconds())
        except (KeyError, ValueError, TypeError):
            started = None
        out.append({
            "label": marker.get("label") or path.stem,
            "pid": marker.get("pid"),
            "started_at": marker.get("started_at"),
            "elapsed_s": elapsed,
        })
    return sorted(out, key=lambda r: r["label"])


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def load_launchd_jobs(log_dir: Path, tail_lines: int = 40) -> list[dict]:
    """launchd*.log(out/err)를 잡별로 묶어 상태 요약. 파일만 읽는다(launchctl 미호출).

    상태는 **out 로그 tail 마커** 기반 추정(최신 런의 완료/실패 신호). 정확한 종료코드가
    아니므로 raw tail 을 함께 노출한다. err 로그는 누적되므로(과거 트레이스백이 남음)
    stderr 존재는 실패가 아니라 '확인 힌트'로만 표시한다. last_run 은 로그 mtime(로컬).
    """
    from datetime import datetime

    if not log_dir.exists():
        return []
    groups: dict[str, dict[str, Path]] = {}
    for p in sorted(log_dir.glob("launchd*.log")):
        stem = p.name[len("launchd"):].removesuffix(".log")  # 예: ".kr.out"
        kind = "err" if stem.endswith(".err") else "out"
        job = stem.removesuffix(".err").removesuffix(".out").strip(".") or "main"
        groups.setdefault(job, {})[kind] = p

    jobs: list[dict] = []
    for job, files in sorted(groups.items()):
        out_p, err_p = files.get("out"), files.get("err")
        out_tail = _tail(out_p, tail_lines) if out_p else []
        err_tail = _tail(err_p, tail_lines) if err_p else []
        mtimes = [p.stat().st_mtime for p in (out_p, err_p) if p and p.exists()]
        last_run = datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds") if mtimes else None
        market_status = _market_statuses(out_tail)
        if market_status:
            vals = set(market_status.values())
            if "error" in vals and "ok" in vals:
                status = "partial"  # 일부 시장만 실패 — 다른 시장은 정상
            elif "error" in vals:
                status = "error"
            else:
                status = "ok"
        else:  # 시장 라인 없는 잡: 범용 마커 폴백
            joined = "\n".join(out_tail)
            if any(m in joined for m in _FAIL_MARKERS):
                status = "error"
            elif any(m in joined for m in _OK_MARKERS):
                status = "ok"
            else:
                status = "unknown"
        jobs.append({
            "job": job, "status": status, "last_run": last_run,
            "markets": market_status,
            "out_tail": out_tail, "err_tail": err_tail,
            "has_stderr": bool(err_tail and any(ln.strip() for ln in err_tail)),
        })
    return jobs


# ── LLM 비용·토큰 (라우터 usage 로그) ──

def usage_report(log_dir: Path) -> dict:
    """USAGE/llm_usage 이벤트 → 일별·모델별 토큰(in/out) 집계.

    토큰은 사실이므로 그대로 집계만 한다 — 달러 환산은 하지 않는다. 단가표는 수시로
    바뀌고(도입가·시간조건·자동캐싱 할인 등) 로컬 추정이 실제 청구와 어긋나므로,
    비용은 각 프로바이더 콘솔에서 확인한다. 모델 키는 provider:model.
    """
    daily: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    total_in = total_out = 0
    for rec in iter_events(log_dir, "USAGE", "llm_usage"):
        day = str(rec.get("ts", ""))[:10]
        tin = int(rec.get("in") or 0)
        tout = int(rec.get("out") or 0)
        model = f'{rec.get("provider")}:{rec.get("model")}'
        d = daily.setdefault(day, {"day": day, "in": 0, "out": 0})
        d["in"] += tin
        d["out"] += tout
        m = by_model.setdefault(model, {"model": model, "in": 0, "out": 0})
        m["in"] += tin
        m["out"] += tout
        total_in += tin
        total_out += tout
    return {
        "daily": [daily[k] for k in sorted(daily)],
        "by_model": sorted(by_model.values(), key=lambda r: r["in"] + r["out"], reverse=True),
        "total_in": total_in,
        "total_out": total_out,
    }


# ── 메모리 스토어 (읽기 전용 표시 경로) ──
#
# 챗 grounding 용 컨텍스트 조립과는 별개 경로다. 저쪽은 LLM 프롬프트에 통째로 들어가는
# 사실원이라 항목이 늘수록 토큰이 불고 인용 정확도가 떨어져서 시장별 최근 몇 건으로
# 잘라야 한다. 사람이 읽는 화면은 그 제약을 받을 이유가 없으므로 store 를 직접 읽는다.
# 쓰기는 하지 않는다 — 승격·retire·importance 는 전부 자동 게이트의 단독 권한이다.


def _query(db_path: Path, market: str, **kw) -> list:
    """메모리 store 를 짧게 열고 닫는다 — 일일 루프의 쓰기와 락을 다투지 않도록."""
    if not Path(db_path).exists():
        return []
    from memory.store import MemoryStore

    store = MemoryStore(db_path)
    try:
        return store.query(market, **kw)
    finally:
        store.close()


def _daily_episodic(entries: list) -> list:
    """주간 회고 엔트리를 제외한 순수 일간 결정 기록."""
    return [e for e in entries if e.data.get("kind") != "weekly_reflection"]


def weekly_reflections(db_path: Path, market: str, today: date) -> list[dict]:
    """주차별 회고 — 기록된 주 + 있어야 했는데 없는 주를 함께 최신순으로.

    회고는 일요일 스텝에서만 생성되므로, 일요일 잡이 걸러진 주는 그대로 유실된다.
    표본 부족(창 안에 결과 기입된 결정이 최소 건수 미만)으로 안 만들어진 주와
    구분해서 표시해야 운용 사고를 정상 동작으로 오해하지 않는다.
    """
    from reflection.weekly import MIN_ENTRIES, WINDOW_DAYS

    entries = _query(db_path, market, store="episodic")
    if not entries:
        return []
    daily = _daily_episodic(entries)
    if not daily:
        return []
    logged = {
        e.day: {
            # content 는 "[{market} 주간 reflection {day}] {요약}" 형태 — 접두를 떼고 본문만
            "summary": e.content.split("] ", 1)[-1],
            "report": e.data.get("report") or {},
        }
        for e in entries
        if e.data.get("kind") == "weekly_reflection"
    }

    first = min(e.day for e in daily)
    sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    rows: list[dict] = []
    while sunday <= today:
        since = sunday - timedelta(days=WINDOW_DAYS)
        eligible = sum(1 for e in daily if e.outcome is not None and since < e.day <= sunday)
        row = {"day": sunday.isoformat(), "eligible": eligible}
        if sunday in logged:
            rows.append({**row, "status": "ok", **logged[sunday]})
        elif eligible >= MIN_ENTRIES:
            rows.append({**row, "status": "missing", "summary": "", "report": {}})
        else:
            rows.append({**row, "status": "insufficient", "summary": "", "report": {}})
        sunday += timedelta(days=7)
    return list(reversed(rows))


def admission_progress(db_path: Path, market: str) -> list[dict]:
    """패턴별 승격 진행도 — 승격 게이트가 실제로 보는 표본을 그대로 재현.

    게이트와 같은 필터(결과 기입됨 · active · 동점 제외 · 이미 승격에 쓰인 표본 제외)를
    쓰므로, "왜 아직 승격이 없는가"를 표본 수와 p 값으로 직접 읽을 수 있다.
    """
    from memory.admission import ALPHA, MIN_N, sign_test_p

    episodic = _daily_episodic(_query(db_path, market, store="episodic", status="active"))
    promoted = _query(db_path, market, store="semantic") + _query(
        db_path, market, store="procedural"
    )
    used = {src for e in promoted for src in e.data.get("source_ids", [])}

    by_pattern: dict[str, list] = {}
    for e in episodic:
        if e.pattern_key:
            by_pattern.setdefault(e.pattern_key, []).append(e)

    rows: list[dict] = []
    for key, group in by_pattern.items():
        # 동점(outcome == 0)은 게이트가 표본에서 빼므로 여기서도 뺀다 — 표시가 게이트와
        # 어긋나면 "n 이 5인데 왜 승격이 안 되나" 같은 오해가 생긴다. 대신 별도 열로 센다.
        scored = [e for e in group if e.id not in used and e.outcome is not None]
        fresh = [e for e in scored if e.outcome != 0]
        ties = len(scored) - len(fresh)
        pending = sum(1 for e in group if e.outcome is None)
        n = len(fresh)
        k_pos = sum(1 for e in fresh if e.outcome > 0)
        mean = sum(e.outcome for e in fresh) / n if n else None
        p = None
        if n:
            p = min(sign_test_p(k_pos, n), sign_test_p(n - k_pos, n))
        if n < MIN_N:
            stage = f"표본 {n}/{MIN_N}"
        elif p is not None and p <= ALPHA:
            stage = "게이트 통과 (다음 스텝에 승격)"
        else:
            stage = f"유의성 미달 (p={p:.3f})"
        rows.append({
            "pattern": key,
            "n": n,
            "ties": ties,
            "pending": pending,
            "k_pos": k_pos,
            "mean_outcome": mean,
            "p_value": p,
            "stage": stage,
        })
    return sorted(rows, key=lambda r: (-r["n"], r["pattern"]))


def promoted_memories(db_path: Path, market: str) -> list[dict]:
    """semantic/procedural 전 상태(probation·active·retired) — 승격 이후의 생애."""
    rows: list[dict] = []
    for store_name in ("semantic", "procedural"):
        for e in _query(db_path, market, store=store_name):
            rows.append({
                "id": e.id,
                "store": store_name,
                "status": e.status,
                "pattern": e.pattern_key,
                "n": e.data.get("n"),
                "p_value": e.data.get("p_value"),
                "mean_outcome": e.data.get("mean_outcome"),
                "probation_until": e.data.get("probation_until"),
                "importance": e.importance,
                "content": e.content,
            })
    return sorted(rows, key=lambda r: (r["status"], r["id"]))


def episodic_ledger(db_path: Path, market: str, limit: int = 60) -> list[dict]:
    """일간 결정 기록 원장 — 최신순. outcome 은 다음 관측에서 소급 기입된다."""
    rows = [
        {
            "day": e.day.isoformat(),
            "pattern": e.pattern_key,
            "outcome": e.outcome,
            "status": e.status,
            "importance": e.importance,
            "cited": " ".join(
                e.data.get("cited_signal_ids", []) + e.data.get("cited_memory_ids", [])
            ),
            "content": e.content,
        }
        for e in _daily_episodic(_query(db_path, market, store="episodic"))
    ]
    return list(reversed(rows))[:limit]


def _veto_verdict(outcome: float | None) -> str:
    """원안의 사후 성과로 본 veto 판정 — 부호가 곧 판정이다."""
    if outcome is None:
        return "대기"
    if outcome > 0:
        return "veto 손해 (원안이 유리했다)"
    if outcome < 0:
        return "veto 적중 (원안이 불리했다)"
    return "동점"


def counterfactual_ledger(db_path: Path, market: str, limit: int = 60) -> list[dict]:
    """하드 veto 로 집행되지 않은 원안 원장 — 최신순. 그 판단이 옳았는지 사후 계측.

    `outcome` 은 원안대로 갔다면의 초과수익, `executed_outcome` 은 대신 동결된 실제
    결정의 초과수익이다. veto 는 배분을 직전 값으로 얼려 행동을 hold 로 만들기 때문에
    막힌 패턴의 표본이 그대로 끊긴다 — 이 기록이 없으면 한 번 선 제약을 무효화할
    증거가 영영 모이지 않는다. 양수가 쌓이면 퇴출 심사가 제약을 푼다.
    """
    entries = _query(db_path, market, store="counterfactual")
    if not entries:
        return []
    executed = {e.id: e.outcome for e in _query(db_path, market, store="episodic")}
    rows = [
        {
            "day": e.day.isoformat(),
            "pattern": e.pattern_key,
            "outcome": e.outcome,
            "executed_outcome": executed.get(e.data.get("executed_id")),
            "verdict": _veto_verdict(e.outcome),
            "content": e.content,
        }
        for e in entries
    ]
    return list(reversed(rows))[:limit]
