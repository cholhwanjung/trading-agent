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

from adapters.base import DISCLOSURE_SOURCES, is_market_closed_error, is_market_weekend
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
            # 정책 텍스트 판본 — 프롬프트·플레이북 개정의 전/후를 로그에서 가른다.
            # None = 판본 기록이 시작되기 전의 결정.
            "prompt_rev": d.get("prompt_rev"),
            # 건강 패널용 — 결정 메타 원본. 필드가 시기별로 추가돼 왔으므로 '키가 없다'
            # (그때는 기록하지 않았다)와 '값이 비었다'를 소비자가 직접 가른다.
            "decision": d,
            "run_day": str(rec.get("ts", ""))[:10],
            "n_bars": rec.get("n_bars") or {},
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

    `tradable` 은 설정상 역할, `executable` 은 **실계좌가 지금 담을 수 있는가**다. 둘을
    가르는 이유는 예산이 없는 날(미입금·조회 실패) 설정값만으로 집행 유니버스를 그리면
    한 주도 못 사는 계좌가 정상 집행 중인 것처럼 보이기 때문이다.
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
        # 설정상 집행 종목이라도 계좌가 그 비중을 표현하지 못하면 실제로 담을 수 없다.
        # 화면의 '집행'을 설정값으로만 그리면 미입금·소액 계좌에서 담기지 않을 종목이
        # 집행 유니버스로 보인다 — 계좌 크기 때문에 좁아진 정의역이 넓어 보이는 셈이다.
        # 판정 근거는 아래 verdict 와 같다(예산 없음·1주 비중 초과는 표현 불가).
        executable = bool(
            block.get("tradable")
            and budget is not None
            and (fractional or (step is not None and step <= cap))
        )
        rows.append({
            "symbol": symbol,
            "asset_class": block.get("asset_class"),
            "index": block.get("index"),  # ETF 만 채워진다
            "tradable": bool(block.get("tradable")),
            "executable": executable,  # 실계좌가 지금 실제로 담을 수 있는가
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

    days_since_run 은 잡이 마지막으로 **돈** 날부터의 간격이다. 주말에는 잡이 돌아도
    결정을 내리지 않고 건너뛴 사실만 남기므로, 결정 신선도만 보면 월요일마다 멀쩡한 잡이
    멈춘 것처럼 보인다. 경고는 이 값으로 걸고 결정 신선도는 표시용으로 남긴다.
    """
    decisions = read_recent_decisions(log_dir, market)
    last = decisions[-1] if decisions else None
    last_day = last["day"] if last else None
    skip_days = [str(rec.get("ts", ""))[:10] for rec in iter_events(log_dir, market, "daily_skip")]
    last_run_day = max([d for d in (last_day, *skip_days) if d], default=None)

    def _age(day: str | None) -> int | None:
        try:
            return (today - date.fromisoformat(day)).days if day else None
        except ValueError:
            return None

    return {
        "market": market,
        "last_day": last_day,
        "days_stale": _age(last_day),
        "days_since_run": _age(last_run_day),
        "skipped_since": bool(last_run_day and last_run_day != last_day),
        "mdd": last.get("mdd") if last else None,
        "violation_days": [d["day"] for d in decisions if d["risk_violations"]][-3:],
    }


def latest_meta_event(log_dir: Path) -> dict | None:
    """META 네임스페이스 최신 meta_shadow 이벤트(제안 가중치·틸트 근거). 없으면 None."""
    events = list(iter_events(log_dir, "META", "meta_shadow"))
    return events[-1] if events else None


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


def load_holding_weights(log_dir: Path, market: str) -> tuple[str, dict[str, float]] | None:
    """마켓 내 **실보유** 배분(CASH 포함) — 마지막으로 체결된 배분과 그 기록일. 없으면 None.

    직전 목표(`risk_{market}.json` 의 prev_weights)를 쓰면 안 된다. 목표는 계좌가 그것을
    표현하지 못한 날에도 갱신되므로, 자금이 없어 한 주도 못 산 시장이 목표대로 담고 있는
    것처럼 그려진다 — 실측으로 KR 은 사흘 연속 체결이 `{CASH: 1.0}` 인 동안 목표가
    `278530` 25% 였다. 체결 배분은 주문 시점의 실제 보유 수량에서 계산되므로 그 혼동이 없다.

    체결이 없던 날에는 이 기록이 남지 않는다 — 그런 날 보유는 바뀌지 않았으므로 마지막
    기록이 곧 현재 보유다. 날짜를 함께 돌려주는 이유는 예산 스냅샷과 같다: 언제 것인지
    없이 보여주면 오래된 보유를 현재 보유로 읽는다.
    """
    last = None
    for rec in iter_events(log_dir, market, "execution_fidelity"):
        executed = rec.get("executed")
        if executed:
            last = (str(rec.get("ts", ""))[:10], executed)
    return last


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
    쓰므로, "왜 아직 승격이 없는가"를 표본 수와 p 값으로 직접 읽을 수 있다. 게이트가
    후보에서 빼는 hold 패턴은 행은 남기고 단계만 '승격 제외'로 적는다.
    """
    from memory.admission import ALPHA, MIN_N, sign_test_p
    from memory.journal import is_hold_pattern

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
        if is_hold_pattern(key):
            # 게이트가 hold 패턴을 표본에서 뺀다 — 통계는 보여 주되 승격 후보로 읽히면 안 된다
            stage = "승격 제외 (hold 패턴)"
        elif n < MIN_N:
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


# ── 에이전트 건강 (성과와 별개로 '시스템이 제대로 돌았는가') ──
#
# 하루 한 번 결정하는 시스템에서는 오류'율'이 성립하지 않는다(표본이 하루 1건). 그래서
# 여기 로더들은 비율이 아니라 **기대한 것이 실제로 거기 있는가**를 센다 — 그날 런이
# 있었는가, 리스크 엔진이 배분을 얼마나 바꿨는가, 목표가 계좌에 닿았는가, 입력 채널이
# 비지 않았는가. 전부 이미 남아 있는 로그·상태 파일에서 계산한다.

RUN_BAD = frozenset({"error", "rejected", "degraded", "missing"})
RUN_WARN = frozenset({"stalled", "late"})


def _run_status(market: str, events: dict[str, list[dict]] | None, is_today: bool) -> dict:
    """하루치 이벤트 → 런 상태. 일간 스텝이 표준출력에만 남기는 분류와 같은 규칙이다.

    브로커 오류 원문은 싣지 않는다 — 요청 주소가 붙어 계좌 식별자가 섞여 나온다.
    """
    if not events:
        # 오늘은 아직 실행 시각 전일 수 있다. 지난 날에 한 줄도 없으면 잡이 돌지 않은 것 —
        # 주말·휴장일에도 잡은 돌아서 건너뛴 사실을 남긴다.
        return {"status": "pending" if is_today else "missing", "detail": ""}
    steps = events.get("daily_step")
    if steps:
        step = steps[-1]
        if step.get("accepted"):
            stalls = events.get("execution_stall")
            if stalls:
                return {"status": "stalled", "detail": str(stalls[-1].get("reason") or "")}
            return {"status": "ok", "detail": ""}
        degraded = step.get("execution_mode") == "degraded"
        try:
            ran_at = datetime.fromisoformat(str(step.get("ts")))
        except ValueError:
            ran_at = None
        weekend = ran_at is not None and not degraded and is_market_weekend(market, ran_at)
        if is_market_closed_error(step.get("error")) or weekend:
            return {"status": "closed", "detail": "주문 거부(장 마감)"}
        return {"status": "degraded" if degraded else "rejected", "detail": ""}
    if events.get("daily_skip"):
        return {"status": "closed", "detail": str(events["daily_skip"][-1].get("reason") or "")}
    errors = events.get("daily_step_error") or []
    return {"status": "error", "detail": str(errors[-1].get("error_type") or "") if errors else ""}


def run_ledger(log_dir: Path, market: str, today: date, days: int = 30) -> list[dict]:
    """최근 days 일의 일간 런 원장 — 하루 한 칸 {day, status, detail}, 오래된 날부터.

    status: ok(주문까지 정상) · stalled(결정은 수리됐으나 한 건도 못 냄) · closed(주말·
    휴장) · degraded(계좌 조회 실패로 실주문 생략) · rejected(장은 열렸는데 거부) ·
    error(스텝이 예외로 끝남) · late(그날 몫의 결정이 날짜를 넘겨 실행됨) · missing(그날
    기록이 한 줄도 없음) · pending(오늘, 아직). 날짜는 기록 시각(UTC)의 날짜 — 로그
    파일명과 같은 축이다.

    late 와 missing 을 가르는 이유: 기기가 자다 깨면 스케줄러가 밀린 런을 뒤늦게 돌린다.
    결정은 났으므로 누락이 아니지만, 예정 시각의 시세가 아니라 몇 시간 뒤 시세로 집행됐다.
    """
    by_day: dict[str, dict[str, list[dict]]] = {}
    for event in ("daily_step", "daily_skip", "daily_step_error", "execution_stall"):
        for rec in iter_events(log_dir, market, event):
            by_day.setdefault(str(rec.get("ts", ""))[:10], {}).setdefault(event, []).append(rec)
    if not by_day:
        return []
    late: dict[str, str] = {}  # 결정 대상일 → 실제로 기록된 시각(날짜를 넘긴 것만)
    for steps in (events.get("daily_step", []) for events in by_day.values()):
        for step in steps:
            asof, ts = str(step.get("asof_day", ""))[:10], str(step.get("ts", ""))
            if asof and ts[:10] > asof:
                late[asof] = ts[:16].replace("T", " ")
    first = min(by_day)  # 기록이 시작되기 전의 날은 누락이 아니다
    rows = []
    for back in range(days - 1, -1, -1):
        day = (today - timedelta(days=back)).isoformat()
        if day < first:
            continue
        if day not in by_day and day in late:
            rows.append({"day": day, "status": "late", "detail": f"{late[day]} UTC 에 실행"})
        else:
            rows.append({"day": day, **_run_status(market, by_day.get(day), is_today=back == 0)})
    return rows


def run_summary(ledger: list[dict], today: date) -> dict:
    """런 원장 요약 — 마지막으로 판정된 런, 마지막 정상(ok) 런의 나이, 상태별 일수."""
    judged = [r for r in ledger if r["status"] != "pending"]
    last_ok = next((r["day"] for r in reversed(judged) if r["status"] == "ok"), None)
    counts: dict[str, int] = {}
    for r in judged:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "last": judged[-1] if judged else None,
        "last_ok_day": last_ok,
        "days_since_ok": (today - date.fromisoformat(last_ok)).days if last_ok else None,
        "counts": counts,
    }


def _l1(a: dict[str, float], b: dict[str, float]) -> float:
    """두 배분의 비중 절대차 합(나누지 않은 L1). 체결 괴리 로그와 같은 정의."""
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def risk_interventions(decisions: list[dict]) -> list[dict]:
    """결정별 리스크 엔진 개입 — LLM 원안(weights_pre_risk) 대비 최종 배분의 거리와 사유.

    l1 이 None 이면 그 결정에는 원안이 기록되지 않았다(초기 로그). rules 는 위반
    문자열의 첫 토큰(max_weight·min_cash·turnover·mdd_circuit·forbidden_*).
    """
    rows = []
    for r in decisions:
        pre = r.get("weights_pre_risk")
        rows.append({
            "day": r["day"],
            "l1": round(_l1(pre, r["weights"]), 6) if pre else None,
            "rules": sorted({v.split()[0] for v in r["risk_violations"] if v}),
            "circuit_open": bool(r["decision"].get("circuit_open")),
            "mdd": r.get("mdd"),
        })
    return rows


def risk_summary(rows: list[dict]) -> dict:
    """개입 요약 — 원안이 기록된 결정 중 배분이 바뀐 비율·크기·사유별 일수."""
    measured = [r for r in rows if r["l1"] is not None]
    changed = [r for r in measured if r["l1"] > 1e-9]
    by_rule: dict[str, int] = {}
    for r in rows:
        for rule in r["rules"]:
            by_rule[rule] = by_rule.get(rule, 0) + 1
    return {
        "n": len(measured),
        "n_changed": len(changed),
        "mean_l1": sum(r["l1"] for r in changed) / len(changed) if changed else None,
        "max_l1": max((r["l1"] for r in changed), default=None),
        "by_rule": by_rule,
    }


def execution_chain(log_dir: Path, market: str) -> list[dict]:
    """목표 → 체결 대사 — 런 날짜별 {intended, executed, drift_l1, dropped, stall, worst_gap}.

    drift_l1 은 목표와 체결 뒤 실보유의 거리, dropped 는 종목별로 목표에 못 닿은 사유
    (거래단위·예산 등), stall 은 주문을 한 건도 못 낸 사유, worst_gap 은 결정에 쓴
    가격과 집행 시점 가격의 최대 괴리(shadow). 해당 기록이 없는 칸은 None.
    """
    rows: dict[str, dict] = {}

    def row(rec: dict) -> dict:
        day = str(rec.get("ts", ""))[:10]
        return rows.setdefault(day, {
            "day": day, "intended": None, "executed": None, "drift_l1": None,
            "dropped": {}, "stall": None, "stall_days": None, "worst_gap": None,
        })

    for rec in iter_events(log_dir, market, "execution_fidelity"):
        row(rec).update(
            intended=rec.get("intended"), executed=rec.get("executed"),
            drift_l1=rec.get("drift_l1"), dropped=rec.get("dropped") or {},
        )
    for rec in iter_events(log_dir, market, "execution_stall"):
        row(rec).update(stall=rec.get("reason"), stall_days=rec.get("days"))
    for rec in iter_events(log_dir, market, "execution_gap"):
        row(rec).update(worst_gap=rec.get("worst_gap"))
    return [rows[d] for d in sorted(rows)]


def load_stall(state_dir: Path, market: str) -> dict | None:
    """현재 집행 정지 상태 {reason, since}. 정지 중이 아니면 reason 이 None. 파일 없으면 None."""
    path = state_dir / f"stall_{market}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def citation_check(decisions: list[dict]) -> list[dict]:
    """결정별 인용 실재 검사 — 근거로 든 ID 가 그날 실제로 제공된 재료 안에 있는가.

    status: cited(전부 제공 목록 안) · unknown(목록 밖 ID 를 인용) · uncited(재료는
    있었는데 인용 없음) · no_material(인용할 신호·교훈이 없던 날) · unrecorded(제공
    목록을 기록하기 전의 결정이라 대조 불가). 인용률은 no_material 을 빼고 읽어야 한다 —
    신호 라이브러리가 없는 시장은 인용할 것이 없어서 0 이지 근거를 안 댄 것이 아니다.
    """
    rows = []
    for r in decisions:
        d = r["decision"]
        cited = set(r["cited_signal_ids"]) | set(r["cited_memory_ids"])
        if "alpha_signals_provided" not in d:
            status, outside = "unrecorded", set()
        else:
            material = set(d.get("alpha_signals_provided") or []) | set(
                d.get("retrieved_memory_ids") or []
            )
            outside = cited - material
            status = (
                "unknown" if outside
                else "cited" if cited
                else "uncited" if material
                else "no_material"
            )
        rows.append({
            "day": r["day"], "status": status, "n_cited": len(cited), "outside": sorted(outside),
        })
    return rows


def revision_spans(decisions: list[dict]) -> list[dict]:
    """판본·모델·설정과 표본 오염 구간을 연속 구간으로 — {field, value, first_day, last_day, n}.

    prompt_rev·config_rev·model 은 값이 바뀔 때마다 새 구간이다(None = 기록 전).
    circuit_open·memory_retrieved 는 참인 구간만 낸다 — 서킷이 배분을 대신 정했거나
    교훈이 결정에 실린 날은 프롬프트·모델이 같아도 다른 조건의 표본이다. 구간 경계가
    곧 시계열에 얹는 변경 표식이다.
    """
    def value(r: dict, field: str):
        d = r["decision"]
        if field == "circuit_open":
            return bool(d.get("circuit_open"))
        if field == "memory_retrieved":
            return bool(d.get("retrieved_memory_ids"))
        return d.get(field)

    spans: list[dict] = []
    for field in ("prompt_rev", "config_rev", "model", "circuit_open", "memory_retrieved"):
        current: dict | None = None
        for r in decisions:
            v = value(r, field)
            if current is not None and current["value"] == v:
                current["last_day"], current["n"] = r["day"], current["n"] + 1
                continue
            current = {"field": field, "value": v, "first_day": r["day"], "last_day": r["day"], "n": 1}
            spans.append(current)
    flags = ("circuit_open", "memory_retrieved")
    return [s for s in spans if s["field"] not in flags or s["value"] is True]


def change_markers(spans: list[dict]) -> list[dict]:
    """구간 경계 → 시계열 변경 표식 [{day, label}]. 각 필드의 첫 구간(기록 시작)은 뺀다."""
    seen: set[str] = set()
    marks = []
    for s in spans:
        flag = s["field"] in ("circuit_open", "memory_retrieved")
        if s["field"] in seen or flag:
            marks.append({"day": s["first_day"], "label": s["field"]})
        seen.add(s["field"])
    return sorted(marks, key=lambda m: m["day"])


# 일간 주기 채널이 이 일수를 넘겨 묵으면 표시한다. 주식은 주말 이틀 + 휴장 하루 + 관측
# 상한(t−1)을 감안한 달력일, 24시간 시장은 그 여유가 필요 없다. 긴 연휴에는 정상인데도
# 켜질 수 있다 — 그때는 런 원장이 closed 로 찍혀 있어 구분된다.
FRESH_MAX_AGE = {"CRYPTO": 2, "US": 4, "KR": 4}


def freshness(state_dir: Path, obs_dir: Path, log_dir: Path, market: str, today: date) -> list[dict]:
    """입력·shadow 채널별 최신 관측의 나이 — [{channel, asof, age_days, daily}].

    daily=True 는 매 런마다 갱신돼야 하는 채널(나이 판정 대상), False 는 주기가 다르거나
    조건부로만 갱신되는 채널(나이만 표시). 원천이 없는 채널은 행을 내지 않는다.
    """
    rows: list[dict] = []

    def add(channel: str, asof, daily: bool) -> None:
        text = str(asof)[:10] if asof else None
        try:
            age = (today - date.fromisoformat(text)).days if text else None
        except ValueError:
            age = None
        rows.append({"channel": channel, "asof": text, "age_days": age, "daily": daily})

    days = list_observation_days(obs_dir, market)
    snap = (load_observation(obs_dir, market, days[0]) or {}) if days else {}
    if days:
        add("관측 스냅샷", days[0], True)
        last_bars = [bars[-1].get("day") for bars in (snap.get("bars") or {}).values() if bars]
        if last_bars:
            add("봉(가장 늦은 종목의 마지막 봉)", min(str(d) for d in last_bars), True)
        published = [n.get("published_at") for n in snap.get("news") or [] if n.get("published_at")]
        if published:
            add("뉴스·공시(최신 발행)", max(str(p) for p in published), False)

    def state(name: str) -> dict:
        path = state_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    regime = state("regime_latest.json").get(market) or {}
    if regime:
        add("국면 FSM", regime.get("asof"), True)
    valuation = (state(f"index_valuation_{market}.json").get("value") or {}).get("asof")
    if valuation:
        add("지수 밸류에이션", valuation, False)
    history = state(f"index_series_{market}.json").get("history") or []
    if history:
        add("지수 시계열", history[-1].get("day") if isinstance(history[-1], dict) else None, True)
    jm = state(f"jm_{market}.json").get("trained_through")
    if jm:
        add("jump model 학습 종료일", jm, False)
    watch = state(f"watch_{market}.json").get("ref_at")
    if watch:
        add("워처 기준가", watch, False)
    return rows


# 채널별로 '매 런마다 채워져 있어야 하는가'. 여기 없는 채널이 0 이면 그냥 빈 것이고(공시가
# 없는 날, 교훈이 없는 시장), 여기 있는 채널이 0 이면 죽은 것이다. 시장마다 다른 이유:
# 가상자산에는 재무·지수 배수가 없고, 순자산가치 괴리는 국내 ETF 만 원천이 있으며,
# 국내 주식은 신호 라이브러리를 열지 않았다.
CHANNEL_EXPECTED: dict[str, frozenset[str]] = {
    "CRYPTO": frozenset({"bars", "features", "news", "budget", "alpha_signals"}),
    "US": frozenset({
        "bars", "features", "news", "budget", "alpha_signals", "fundamentals", "index_valuation",
    }),
    "KR": frozenset({
        "bars", "features", "news", "budget", "fundamentals", "index_valuation", "etf_premium",
    }),
}
CHANNELS = (
    "bars", "features", "news", "disclosures", "news_injected", "disclosures_injected",
    "fundamentals", "index_valuation", "etf_premium", "alpha_signals", "budget", "lessons",
    "upcoming_events",
)


def channel_grid(log_dir: Path, obs_dir: Path, market: str, days: int = 14) -> list[dict]:
    """입력 채널 도달 격자 — 최근 days 결정마다 채널별 건수. None = 그 결정에는 기록이 없다.

    bars·features 는 값이 있는 종목 수, news·disclosures 는 **수집** 건수(관측 스냅샷),
    news_injected·disclosures_injected 는 프롬프트에 **실제로 실린** 건수다. 수집됐다고
    실린 것이 아니다 — 프롬프트는 건수 상한에서 자르므로 둘을 따로 본다. 실린 건수를
    기록하지 않은 결정은 None 으로 남긴다(수집 건수로 메우지 않는다).
    """
    rows = []
    for r in read_recent_decisions(log_dir, market, limit=days):
        d = r["decision"]
        snap = load_observation(obs_dir, market, r["day"]) or {}
        news = snap.get("news")
        injected = d.get("prompt_channels") or {}

        def size(key: str) -> int | None:
            return len(d.get(key) or ()) if key in d else None

        def present(key: str) -> int | None:
            return int(d.get(key) is not None) if key in d else None

        rows.append({
            "day": r["day"],
            "bars": sum(1 for n in r["n_bars"].values() if n) if r["n_bars"] else None,
            "features": (
                sum(1 for v in (d.get("features") or {}).values() if v is not None)
                if "features" in d else None
            ),
            "news": (
                sum(1 for n in news if n.get("source") not in DISCLOSURE_SOURCES)
                if news is not None else None
            ),
            "disclosures": (
                sum(1 for n in news if n.get("source") in DISCLOSURE_SOURCES)
                if news is not None else None
            ),
            "news_injected": injected.get("news"),
            "disclosures_injected": injected.get("disclosures"),
            "fundamentals": size("fundamentals"),
            "index_valuation": present("index_valuation"),
            "etf_premium": size("etf_premium"),
            "alpha_signals": size("alpha_signals_provided"),
            "budget": present("budget"),
            "lessons": size("retrieved_memory_ids"),
            "upcoming_events": size("upcoming_events"),
        })
    return rows


def dead_channels(grid: list[dict], market: str) -> list[dict]:
    """격자에서 죽은 칸만 — 채워져 있어야 하는 채널이 0 인 (day, channel)."""
    expected = CHANNEL_EXPECTED.get(market, frozenset())
    return [
        {"day": row["day"], "channel": ch}
        for row in grid
        for ch in CHANNELS
        if ch in expected and row.get(ch) == 0
    ]


# ── 에이전트 행동·평가 (건강 탭 후반) ──


def usage_breakdown(log_dir: Path) -> dict:
    """llm_usage → 목적별·시장별 호출 수와 토큰. 어디에 토큰이 쓰이는지를 가른다."""
    by_purpose: dict[str, dict] = {}
    by_market: dict[str, dict] = {}
    for rec in iter_events(log_dir, "USAGE", "llm_usage"):
        tin, tout = int(rec.get("in") or 0), int(rec.get("out") or 0)
        for table, key in (
            (by_purpose, rec.get("purpose") or "(미상)"),
            (by_market, rec.get("for_market") or "(공통)"),
        ):
            row = table.setdefault(key, {"key": key, "calls": 0, "in": 0, "out": 0})
            row["calls"] += 1
            row["in"] += tin
            row["out"] += tout

    def ranked(table: dict[str, dict]) -> list[dict]:
        return sorted(table.values(), key=lambda r: r["in"] + r["out"], reverse=True)

    return {"by_purpose": ranked(by_purpose), "by_market": ranked(by_market)}


def stage_moves(decisions: list[dict]) -> list[dict]:
    """결정별 단계 이동량 — 교훈 블렌딩·토론 재결정·리스크 적용이 배분을 얼마나 옮겼는가.

    값은 나누지 않은 L1. 0 = 그 단계가 있었지만 배분을 바꾸지 않았다(교훈 미적용·토론
    미소집 포함), None = 그 결정에는 해당 기록이 없다. 효과가 아니라 **개입의 크기**다 —
    옮긴 것이 옳았는지는 여기서 알 수 없다. tokens_in 은 그 결정의 입력 토큰 합.
    """
    rows = []
    for r in decisions:
        d = r["decision"]
        influence, debate = d.get("influence") or {}, d.get("debate") or {}
        memory = 0.0 if "influence" in d else None
        if influence.get("applied"):
            # 블렌딩은 두 배분의 볼록 결합이라 이동량은 배율 × 두 배분의 거리다
            memory = influence["scale"] * _l1(influence["base_weights"], influence["mem_weights"])
        moved = 0.0 if "debate" in d else None
        if debate.get("proposed_before") and debate.get("allocation_after"):
            moved = _l1(debate["proposed_before"], debate["allocation_after"])
        pre = r.get("weights_pre_risk")
        rows.append({
            "day": r["day"],
            "memory": None if memory is None else round(memory, 6),
            "debate": None if moved is None else round(moved, 6),
            "risk": round(_l1(pre, r["weights"]), 6) if pre else None,
            "debated": bool(debate),
            "tokens_in": (d.get("tokens") or {}).get("in"),
        })
    return rows


def memory_ladder(decisions: list[dict]) -> dict:
    """교훈이 결정에 닿는 사다리 — 회수됨 → 배분에 반영됨 → 근거로 인용됨(결정 수).

    단마다 줄어드는 것이 정상이고, 어느 단에서 끊기는지가 정보다: 회수가 0 이면 쓸 교훈이
    없는 것이고, 회수는 되는데 반영이 0 이면 신뢰도 게이트가 막은 것이다.
    """
    recorded = [r for r in decisions if "retrieved_memory_ids" in r["decision"]]
    return {
        "n": len(recorded),
        "retrieved": sum(1 for r in recorded if r["decision"].get("retrieved_memory_ids")),
        "applied": sum(
            1 for r in recorded if (r["decision"].get("influence") or {}).get("applied")
        ),
        "cited": sum(1 for r in recorded if r["cited_memory_ids"]),
    }


def eval_runs(runs_dir: Path) -> list[dict]:
    """오프라인 평가 실행 목록 — 오래된 것부터. 실행 폴더의 요약·매니페스트만 읽는다."""
    rows = []
    if not runs_dir.exists():
        return rows
    for run in sorted(p for p in runs_dir.iterdir() if (p / "summary.json").exists()):
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        manifest_path = run / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
        rows.append({
            "run_id": run.name,
            "started_at": str(manifest.get("started_at") or "")[:16].replace("T", " "),
            "calls": summary.get("calls"),
            "passed": summary.get("passed"),
            "gates": summary.get("gates") or {},
            "latency_p50_s": summary.get("latency_p50_s"),
            "latency_p95_s": summary.get("latency_p95_s"),
            "categories": {
                k: v.get("pass_rate") for k, v in (summary.get("categories") or {}).items()
            },
            "prompt_rev": manifest.get("chat_prompt_rev"),
            "git_dirty": manifest.get("git_dirty"),
        })
    return rows


# 결정 레코드가 사후에 재구성되려면 있어야 하는 필드 — 무엇을 보고(features), 어떤 판본·
# 모델·설정으로 결정했고, 리스크 엔진 앞의 원안이 무엇이었는가.
RECORD_REQUIRED = ("features", "model", "prompt_rev", "config_rev", "weights_pre_risk")


def scorecard(
    ledger: list[dict], decisions: list[dict], risk_rows: list[dict],
    grid: list[dict], dead: list[dict], citations: list[dict],
) -> list[dict]:
    """결정론 점검표 — 점검마다 {check, bad, total, last_bad}. 판정자 없이 규칙만으로 센다.

    비율이 아니라 **위반 건수와 마지막 위반일**을 낸다 — 하루 한 건짜리 표본에서 비율은
    한 번의 실패로 크게 흔들려 추세를 말해 주지 못한다. 점검마다 창이 다르므로(런 원장은
    달력일, 나머지는 결정 수) total 을 함께 읽는다.
    """
    def row(check: str, bad_days: list[str], total: int) -> dict:
        return {"check": check, "bad": len(bad_days), "total": total,
                "last_bad": max(bad_days) if bad_days else None}

    judged = [r for r in ledger if r["status"] != "pending"]
    accepted = [r for r in judged if r["status"] in ("ok", "stalled")]
    # 서킷이 열린 날은 원안과 무관하게 엔진이 배분을 대신 정한다 — 원안의 한도 준수를 묻는
    # 점검에서 뺀다.
    measured = [r for r in risk_rows if r["l1"] is not None and not r["circuit_open"]]
    # 필드는 시기별로 추가돼 왔다. 처음부터 없던 것은 결손이 아니고, **한 번 기록되기
    # 시작한 필드가 다시 비면** 결손이다.
    seen: set[str] = set()
    incomplete = []
    for r in decisions:
        present = {k for k in RECORD_REQUIRED if r["decision"].get(k) is not None}
        if seen - present:
            incomplete.append(r["day"])
        seen |= present
    with_material = [c for c in citations if c["status"] in ("cited", "unknown", "uncited")]
    cited = [c for c in citations if c["status"] in ("cited", "unknown")]
    return [
        row("런이 제때 돌았다",
            [r["day"] for r in judged if r["status"] in RUN_BAD or r["status"] == "late"],
            len(judged)),
        row("주문이 나갔다", [r["day"] for r in accepted if r["status"] == "stalled"], len(accepted)),
        row("LLM 원안이 한도 안이었다", [r["day"] for r in measured if r["l1"] > 1e-9], len(measured)),
        row("입력 채널이 찼다", sorted({d["day"] for d in dead}), len(grid)),
        row("인용한 ID 가 실재했다", [c["day"] for c in cited if c["status"] == "unknown"], len(cited)),
        row("재료가 있으면 인용했다",
            [c["day"] for c in with_material if c["status"] == "uncited"], len(with_material)),
        row("결정 레코드가 완전했다", incomplete, len(decisions)),
    ]
