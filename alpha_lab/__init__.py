"""Alpha Lab — 신호 self-improve: writer→judge→백테스트→admission."""

from alpha_lab.backtest import BacktestResult, ICStats, run_backtest, score_correlation
from alpha_lab.dsl import DSL_SPEC, DSLError, evaluate, validate
from alpha_lab.library import FactorCandidate, FactorLibrary, FactorRecord
from alpha_lab.loop import generate_candidates

__all__ = [
    "DSL_SPEC",
    "BacktestResult",
    "DSLError",
    "FactorCandidate",
    "FactorLibrary",
    "FactorRecord",
    "ICStats",
    "evaluate",
    "generate_candidates",
    "run_backtest",
    "score_correlation",
    "validate",
]
