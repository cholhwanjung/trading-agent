"""일간 reflection 1단계 — 결정의 episodic 기록 + 결과(행동 vs 무행동) 소급 기입.

outcome 정의: 실제 행동 수익 − 무행동(직전 배분 유지) 수익.
"안 사도 올랐다"를 "잘한 매수"와 구분하는 기여 계측. 단일 값은 noisy 하므로
개별 outcome 은 admission 게이트의 표본일 뿐 그 자체로 승격 근거가 아니다.

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
    entry_id = store.add(
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
    _record_counterfactual(store, market, day, prev_weights, features, decision_meta,
                           prices, entry_id)
    return entry_id


def _record_counterfactual(
    store: MemoryStore,
    market: str,
    day: date,
    prev_weights: dict[str, float] | None,
    features: dict[str, dict | None],
    decision_meta: dict,
    prices: dict[str, float],
    executed_id: str,
) -> None:
    """veto 로 집행되지 않은 원안을 가상 결과 계측용으로 남긴다.

    하드 veto 는 배분을 직전 값으로 동결하므로 집행 기록의 행동 성분이 hold 로 바뀐다.
    그러면 veto 된 패턴(risk_on/risk_off)의 표본이 다시는 쌓이지 않아, 그 패턴을 무효화할
    증거를 제약 자신이 차단한다 — 한 번 선 제약이 영구화되는 구조다. 집행은 그대로 막되
    원안의 가상 성과만 계속 계측해 재검증 루프를 닫는다.
    """
    key = decision_meta.get("counterfactual_key")
    weights = decision_meta.get("weights_pre_risk")
    if not (key and weights):
        return
    store.add(
        market,
        "counterfactual",
        day,
        f"[{market} {day}] {key} — veto 된 원안 "
        f"{ {k: round(v, 2) for k, v in weights.items()} } (미집행)",
        data={
            "weights": weights,
            "prev_weights": prev_weights,
            "prices": prices,
            "features": features,
            "executed_id": executed_id,
        },
        pattern_key=key,
    )


def record_unexecuted(
    store: MemoryStore,
    market: str,
    day: date,
    weights: dict[str, float],
    prev_weights: dict[str, float] | None,
    features: dict[str, dict | None],
    decision_meta: dict,
    prices: dict[str, float],
    reason: str,
) -> str | None:
    """집행이 불가능했던 날의 원안을 counterfactual 로 남긴다 — episodic 이 아니다.

    계좌에 자금이 없거나 1 건 상한에 막혀 한 주도 담지 못한 날, 체결 배분은 전액 현금으로
    나온다. 그것을 episodic 으로 남기면 '에이전트가 현금을 선택했다'는 기록이 되어 내리지
    않은 결정에 성적이 매겨지고, 다음 날 직전 배분까지 그 값으로 오염된다. 배분과 기준가만
    있으면 '그 배분이었다면'의 초과수익은 정의되므로 신호 자체는 반사실로 보존한다.

    episodic 을 남기지 않으므로 직전 배분 사슬은 마지막으로 **실제 집행된** 날에 머문다.
    그 사이가 비는 건 맞지만, 없던 결정을 지어내는 것보다 낫다.
    """
    if store.query(market, store="counterfactual", day=day):
        return None
    key = pattern_key(features, weights, prev_weights)
    return store.add(
        market,
        "counterfactual",
        day,
        f"[{market} {day}] {key} — 미집행({reason}) 원안 "
        f"{ {k: round(v, 2) for k, v in weights.items()} }",
        data={
            "weights": weights,
            "prev_weights": prev_weights,
            "prices": prices,
            "features": features,
            "unexecuted": reason,
            "cited_signal_ids": decision_meta.get("cited_signal_ids", []),
            "cited_memory_ids": decision_meta.get("cited_memory_ids", []),
        },
        pattern_key=key,
    )


def fill_pending_outcomes(
    store: MemoryStore, market: str, prices_now: dict[str, float], today: date
) -> list[tuple[str, float]]:
    """outcome 미기입 기록에 (행동 − 무행동) 수익 차이를 소급 기입.

    entry.prices = 결정 당시 t-1 종가, prices_now = 현재 t-1 종가 → 보유 구간 수익률.
    prev_weights 가 없으면(첫 결정) 무행동 기준이 없어 절대 초과수익 대신 0 대비로 기록.
    veto 된 원안(counterfactual)도 같은 산식으로 계측한다 — 배분과 기준가만 있으면
    되고, 집행 여부와 무관하게 "그 배분이었다면"의 초과수익이 정의된다.
    """
    filled: list[tuple[str, float]] = []
    entries = [
        e
        for name in ("episodic", "counterfactual")
        for e in store.query(market, store=name, outcome_missing=True)
    ]
    for entry in entries:
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
