"""대시보드 데이터 로더 — 순수 stdlib(pandas/streamlit 미의존, 테스트 가능).

일일 루프가 남긴 관측 스냅샷(`data/state/observations/`)과 결정 로그
(`data/logs/`)를 읽어 UI 가 바로 쓰는 평범한 dict/list 로 정규화한다.
브로커 API 를 치지 않고, 파일만 읽는다(결정론·감사 가능).
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
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
        }
    return [rows[k] for k in sorted(rows)][-limit:]


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
