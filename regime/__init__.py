"""시장 국면(regime) 분류 — O'Neil market pulse 이식 ([ADR-023]). 무료 일간 지수 봉만."""

from regime.live import INDEX_PROXY, compute_regime
from regime.meta import MarketSignal, MetaProposal, propose_meta_weights
from regime.pulse import (
    CORRECTION,
    UNDER_PRESSURE,
    UPTREND,
    RegimeResult,
    classify_regime,
)
from regime.state import load_market_signals, update_regime_signal

__all__ = [
    "CORRECTION",
    "UNDER_PRESSURE",
    "UPTREND",
    "RegimeResult",
    "classify_regime",
    "INDEX_PROXY",
    "compute_regime",
    "MarketSignal",
    "MetaProposal",
    "propose_meta_weights",
    "load_market_signals",
    "update_regime_signal",
]
