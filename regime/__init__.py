"""시장 국면(regime) 분류 — O'Neil market pulse 이식. 무료 일간 지수 봉만."""

from regime.jm import JumpFeatures, compute_jump_features, feature_matrix
from regime.jump_model import BEAR, BULL, JumpModel, fit, infer, needs_refit
from regime.live import (
    INDEX_PROXY,
    compute_jm_features,
    compute_jm_regime,
    compute_macro_regime,
    compute_market_vol,
    compute_regime,
)
from regime.macro import CALM, CRISIS, STRESS, MacroRegime, classify_macro
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
    "CALM",
    "STRESS",
    "CRISIS",
    "MacroRegime",
    "classify_macro",
    "compute_macro_regime",
    "INDEX_PROXY",
    "compute_regime",
    "compute_market_vol",
    "JumpFeatures",
    "compute_jump_features",
    "compute_jm_features",
    "BULL",
    "BEAR",
    "JumpModel",
    "compute_jm_regime",
    "feature_matrix",
    "fit",
    "infer",
    "needs_refit",
    "MarketSignal",
    "MetaProposal",
    "propose_meta_weights",
    "load_market_signals",
    "update_regime_signal",
]
