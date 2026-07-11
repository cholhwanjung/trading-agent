"""일간 reflection 1단계 — 결정의 episodic 기록 + 결과(행동 vs 무행동) 소급 기입.

outcome 정의 (설계 §3.5): 실제 행동 수익 − 무행동(직전 배분 유지) 수익.
"안 사도 올랐다"를 "잘한 매수"와 구분하는 기여 계측. 단일 값은 noisy 하므로
개별 outcome 은 admission 게이트의 표본일 뿐 그 자체로 승격 근거가 아니다 (ADR-004).

pattern_key v1: feature 버킷의 결정론 태그 — rsi(과매도/중립/과열) · macd 부호 ·
60d 낙폭 깊이 · 행동(현금 증감). 반복 관측 카운트의 단위가 된다.
"""

from __future__ import annotations

from datetime import date

from adapters.allocation import CASH
from memory.store import MemoryStore


def _bucket_rsi(v: float) -> str:
    return "low" if v < 35 else ("high" if v > 65 else "mid")


def _bucket_dd(v: float) -> str:
    return "deep" if v < -0.15 else ("shallow" if v < -0.05 else "none")


def pattern_key(
    features: dict[str, dict | None],
    weights: dict[str, float],
    prev_weights: dict[str, float] | None,
) -> str:
    """시장 상태 × 행동의 결정론 태그. feature 없으면 상태 미상(unk) 처리."""
    valid = [f for f in features.values() if f]
    if valid:
        rsi = sum(f["rsi_14"] for f in valid) / len(valid)
        macd = sum(f["macd_hist"] for f in valid) / len(valid)
        dd = min(f["drawdown_60d"] for f in valid)
        state = f"rsi={_bucket_rsi(rsi)}|macd={'pos' if macd > 0 else 'neg'}|dd={_bucket_dd(dd)}"
    else:
        state = "rsi=unk|macd=unk|dd=unk"

    cash_now = weights.get(CASH, 0.0)
    cash_prev = (prev_weights or {}).get(CASH, cash_now)
    if cash_now < cash_prev - 0.05:
        action = "risk_on"
    elif cash_now > cash_prev + 0.05:
        action = "risk_off"
    else:
        action = "hold"
    return f"{state}|action={action}"


def record_decision(
    store: MemoryStore,
    market: str,
    day: date,
    weights: dict[str, float],
    prev_weights: dict[str, float] | None,
    features: dict[str, dict | None],
    decision_meta: dict,
    prices: dict[str, float],
    embedding: list[float] | None = None,
) -> str | None:
    """당일 결정을 episodic 으로 기록. 같은 (market, day) 는 멱등(스킵, None 반환)."""
    if store.query(market, store="episodic", day=day):
        return None
    key = pattern_key(features, weights, prev_weights)
    content = (
        f"[{market} {day}] {key} — 배분 { {k: round(v, 2) for k, v in weights.items()} }. "
        f"근거: {decision_meta.get('rationale', '')[:200]}"
    )
    return store.add(
        market,
        "episodic",
        day,
        content,
        data={
            "weights": weights,
            "prev_weights": prev_weights,
            "prices": prices,
            "features": features,
            "cited_signal_ids": decision_meta.get("cited_signal_ids", []),
            "cited_memory_ids": decision_meta.get("cited_memory_ids", []),
            "scenario_invalidation": decision_meta.get("scenario_invalidation", ""),
            "risk_violations": decision_meta.get("risk_violations", []),
        },
        pattern_key=key,
        embedding=embedding,
    )


def fill_pending_outcomes(
    store: MemoryStore, market: str, prices_now: dict[str, float], today: date
) -> list[tuple[str, float]]:
    """outcome 미기입 episodic 에 (행동 − 무행동) 수익 차이를 소급 기입.

    entry.prices = 결정 당시 t-1 종가, prices_now = 현재 t-1 종가 → 보유 구간 수익률.
    prev_weights 가 없으면(첫 결정) 무행동 기준이 없어 절대 초과수익 대신 0 대비로 기록.
    """
    filled: list[tuple[str, float]] = []
    for entry in store.query(market, store="episodic", outcome_missing=True):
        if entry.day >= today:
            continue  # 아직 다음 관측이 없다
        entry_prices = entry.data.get("prices") or {}
        weights = entry.data.get("weights") or {}
        prev = entry.data.get("prev_weights") or {}
        returns = {
            s: prices_now[s] / p0 - 1.0
            for s, p0 in entry_prices.items()
            if s in prices_now and p0
        }
        if not returns:
            continue
        r_action = sum(weights.get(s, 0.0) * r for s, r in returns.items())
        r_hold = sum(prev.get(s, 0.0) * r for s, r in returns.items())
        outcome = r_action - r_hold
        store.update(entry.id, outcome=outcome)
        filled.append((entry.id, outcome))
    return filled
