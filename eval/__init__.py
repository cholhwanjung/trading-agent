"""Eval — 병행 페이퍼 운용·ablation 측정 + 상위 결합 지수."""

from eval.index_bench import (
    index_hist,
    index_path,
    load_index_series,
    normalized,
    record_index_series,
)
from eval.meta import (
    combined_index,
    combined_index_dynamic,
    load_arm_history,
    load_meta_shadow,
    max_drawdown,
    record_meta_shadow,
)
from eval.paper_portfolio import VirtualPortfolio
from eval.perf import daily_returns, drawdown_series, perf_stats
from eval.regime_eval import RegimeScore, compare_regimes
from eval.rolling import meta_rolling_report, meta_shadow_delta, rolling_delta, rolling_report

__all__ = [
    "RegimeScore",
    "VirtualPortfolio",
    "compare_regimes",
    "combined_index",
    "combined_index_dynamic",
    "daily_returns",
    "drawdown_series",
    "index_hist",
    "index_path",
    "load_arm_history",
    "load_index_series",
    "load_meta_shadow",
    "max_drawdown",
    "meta_rolling_report",
    "meta_shadow_delta",
    "normalized",
    "perf_stats",
    "record_index_series",
    "record_meta_shadow",
    "rolling_delta",
    "rolling_report",
]
