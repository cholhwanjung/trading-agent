"""대시보드 데이터 로더 — 순수 stdlib(pandas/streamlit 미의존, 테스트 가능).

일일 루프가 남긴 관측 스냅샷(`data/state/observations/`)과 결정 로그
(`data/logs/`)를 읽어 UI 가 바로 쓰는 평범한 dict/list 로 정규화한다.
브로커 API 를 치지 않고, 파일만 읽는다(결정론·감사 가능).
"""

from __future__ import annotations

import json
from datetime import date
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

_OK_MARKERS = ("briefing=", "status=ok")
_FAIL_MARKERS = ("status=fail", "status=error")


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def load_launchd_jobs(log_dir: Path, tail_lines: int = 20) -> list[dict]:
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
        joined = "\n".join(out_tail)
        if any(m in joined for m in _FAIL_MARKERS):
            status = "error"
        elif any(m in joined for m in _OK_MARKERS):
            status = "ok"
        else:
            status = "unknown"
        jobs.append({
            "job": job, "status": status, "last_run": last_run,
            "out_tail": out_tail, "err_tail": err_tail,
            "has_stderr": bool(err_tail and any(ln.strip() for ln in err_tail)),
        })
    return jobs


# ── LLM 비용·토큰 (라우터 usage 로그) ──

# 프로바이더 단가 (USD / 100만 토큰) — 공식 가격 페이지 기준 스타터 표.
# 마지막 확인 2026-07-24 (platform.claude.com · ai.google.dev). 공지 단가는 수시로
# 바뀌고 시간 조건도 있어(예: Sonnet 5 도입가 $2/$10 는 2026-08-31 까지, 이후 $3/$15)
# data/state/llm_pricing.json 에 {"provider:model": {"in": x, "out": y}} 로 덮어쓴다.
# 토큰은 사실이라 로그에 저장하지만 비용은 단가가 변하므로 여기서 파생한다.
DEFAULT_LLM_PRICING: dict[str, dict[str, float]] = {
    # Anthropic (기본 라우팅 프로바이더)
    "anthropic:claude-sonnet-5": {"in": 2.0, "out": 10.0},  # 도입가, 2026-09-01 부터 3/15
    "anthropic:claude-haiku-4-5-20251001": {"in": 1.0, "out": 5.0},
    "anthropic:claude-opus-4-8": {"in": 5.0, "out": 25.0},
    # Gemini (대체 프로바이더 예시)
    "gemini:gemini-2.5-pro": {"in": 1.25, "out": 10.0},
    "gemini:gemini-2.5-flash": {"in": 0.30, "out": 2.50},
}


def load_pricing(state_dir: Path) -> dict[str, dict[str, float]]:
    """기본 단가에 {state_dir}/llm_pricing.json 오버라이드를 병합한다(없으면 기본만)."""
    pricing = dict(DEFAULT_LLM_PRICING)
    path = state_dir / "llm_pricing.json"
    if path.exists():
        pricing.update(json.loads(path.read_text(encoding="utf-8")))
    return pricing


def _rate(price: object) -> tuple[float, float] | None:
    """단가 항목에서 (in, out) 숫자쌍을 뽑는다. 미설정·비숫자(채워넣기 전 null 자리표시자·
    손편집 오타 포함)면 None → 호출부가 '단가 미등록'으로 처리한다."""
    if isinstance(price, dict):
        pin, pout = price.get("in"), price.get("out")
        if isinstance(pin, (int, float)) and isinstance(pout, (int, float)):
            return float(pin), float(pout)
    return None


def usage_cost_report(log_dir: Path, pricing: dict[str, dict[str, float]]) -> dict:
    """USAGE/llm_usage 이벤트 → 일별·목적별 토큰/비용 집계.

    비용은 pricing[provider:model] 단가로 파생. 단가 미등록 모델은 토큰만 집계하고
    unpriced 에 표시(비용 0 기여) — 사용자가 단가를 채우면 자동 반영된다.
    """
    daily: dict[str, dict] = {}
    by_purpose: dict[str, dict] = {}
    unpriced: set[str] = set()
    total_in = total_out = 0
    total_cost = 0.0
    for rec in iter_events(log_dir, "USAGE", "llm_usage"):
        day = str(rec.get("ts", ""))[:10]
        tin = int(rec.get("in") or 0)
        tout = int(rec.get("out") or 0)
        key = f'{rec.get("provider")}:{rec.get("model")}'
        rate = _rate(pricing.get(key))
        cost = (tin / 1e6 * rate[0] + tout / 1e6 * rate[1]) if rate else 0.0
        if rate is None:
            unpriced.add(key)
        purpose = rec.get("purpose") or "(미상)"
        d = daily.setdefault(day, {"day": day, "in": 0, "out": 0, "cost": 0.0})
        d["in"] += tin
        d["out"] += tout
        d["cost"] += cost
        p = by_purpose.setdefault(purpose, {"purpose": purpose, "in": 0, "out": 0, "cost": 0.0})
        p["in"] += tin
        p["out"] += tout
        p["cost"] += cost
        total_in += tin
        total_out += tout
        total_cost += cost
    return {
        "daily": [daily[k] for k in sorted(daily)],
        "by_purpose": sorted(by_purpose.values(), key=lambda r: r["cost"], reverse=True),
        "total_in": total_in,
        "total_out": total_out,
        "total_cost": total_cost,
        "unpriced": sorted(unpriced),
    }
