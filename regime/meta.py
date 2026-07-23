"""시장 간 shadow 메타 배분 제안 — regime 기반 결정론 틸트 ([ADR-025]).

고정 1/N anchor 대비 각 시장의 국면으로 예산을 bounded 틸트한다. LLM 미개입(순수 함수).
증거 부족(regime_state=None)한 시장은 anchor 유지(무개입 fallback). v1 은 shadow —
제안만 반환하고 집행/Risk 변조는 검증 후(하드룰 1). 출력 weights 는
`treasury.guard.plan_transfers` 의 target_split 로 직결된다(시장 = venue).

regime 이 모두 동일하면 틸트가 균일→재정규화로 anchor 복귀: 시장 간 *상대* 국면차가
없으면 예산을 옮길 이유가 없다(하방 방어는 각 버킷 내부 현금비중이 담당, 격리 유지 [ADR-007]).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from regime.pulse import CORRECTION, UNDER_PRESSURE, UPTREND

# regime → 틸트 부호·강도(max_tilt 배). None/미지 → 0 (무개입 fallback).
REGIME_SCORE: dict[str, float] = {
    UPTREND: 1.0,
    UNDER_PRESSURE: -0.5,
    CORRECTION: -1.0,
}


@dataclass(frozen=True)
class MarketSignal:
    """시장 1곳의 집계 신호 — 격리 유지: 개별 포지션·메모리 미포함 ([ADR-007])."""

    market: str
    regime_state: str | None
    drawdown: float = 0.0
    trail_return_k: float | None = None  # 참고 신호(v1 틸트 미반영 — flow-chasing 회피)


@dataclass(frozen=True)
class MetaProposal:
    asof_day: date | None
    weights: dict[str, float]       # {market: w}, ∑=1, long-only — plan_transfers target_split
    anchor: dict[str, float]        # 비교 기준(균등)
    deviation_l1: float             # 0.5·Σ|w−anchor| (turnover 대칭, 감사용)
    cited_signals: dict[str, str]   # {market: "CORRECTION→감액"} credit assignment
    note: str = ""


def _equal_anchor(markets: list[str]) -> dict[str, float]:
    w = 1.0 / len(markets)
    return {m: w for m in markets}


def propose_meta_weights(
    signals: list[MarketSignal],
    anchor: dict[str, float] | None = None,
    max_tilt: float = 0.10,
    asof_day: date | None = None,
) -> MetaProposal:
    """regime 기반 결정론 틸트. 순수 함수 — I/O 없음.

    각 시장 raw = anchor + score(regime)·max_tilt → long-only clamp → ∑=1 재정규화.
    regime_state=None 시장은 score 0 → anchor 유지(무개입 fallback).
    """
    markets = [s.market for s in signals]
    if not markets:
        return MetaProposal(asof_day, {}, {}, 0.0, {}, "no markets")
    anchor = anchor or _equal_anchor(markets)

    cited: dict[str, str] = {}
    raw: dict[str, float] = {}
    for s in signals:
        score = REGIME_SCORE.get(s.regime_state or "", 0.0)
        tilt = score * max_tilt
        raw[s.market] = max(0.0, anchor.get(s.market, 0.0) + tilt)
        label = "증액" if tilt > 0 else "감액" if tilt < 0 else "유지"
        cited[s.market] = f"{s.regime_state or 'none'}→{label}"

    total = sum(raw.values())
    weights = {m: (w / total if total > 0 else anchor.get(m, 0.0)) for m, w in raw.items()}
    dev = 0.5 * sum(abs(weights[m] - anchor.get(m, 0.0)) for m in weights)
    note = "regime tilt" if dev > 1e-9 else "anchor (uniform/no signal)"
    return MetaProposal(asof_day, weights, dict(anchor), round(dev, 4), cited, note)
