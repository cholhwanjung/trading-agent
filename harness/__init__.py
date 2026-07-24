"""Paper trading harness — 어댑터 + baseline 정책 + 구조화 로깅."""

from harness.deadline import DEFAULT_DEADLINE_S, run_deadline_s, with_deadline
from harness.env import load_env
from harness.jsonlog import JsonlLogger, iter_events
from harness.loop import AllocationError, run_daily_step, validate_weights
from harness.netgate import wait_for_network
from harness.notify import notify
from harness.policy import CASH, BuyAndHold, Policy, RandomPolicy
from harness.runner import MarketRun, run_all_markets

__all__ = [
    "CASH",
    "DEFAULT_DEADLINE_S",
    "AllocationError",
    "BuyAndHold",
    "JsonlLogger",
    "iter_events",
    "load_env",
    "MarketRun",
    "notify",
    "Policy",
    "RandomPolicy",
    "run_all_markets",
    "run_daily_step",
    "run_deadline_s",
    "validate_weights",
    "wait_for_network",
    "with_deadline",
]
