"""3시장 동시 러너 — 시장별 독립 실행 + 실패 격리.

한 시장의 장애(API 다운, 누출 감지)가 다른 시장의 스텝을 막지 않는다.
실패는 구조화 이벤트(daily_step_error)로 기록되고 결과 dict에 예외로 담긴다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date

from adapters.base import MarketAdapter, OrderResult
from harness.jsonlog import JsonlLogger
from harness.loop import run_daily_step
from harness.policy import Policy


@dataclass(frozen=True)
class MarketRun:
    """시장 1개의 실행 단위: 어댑터 + 정책 + 유니버스."""

    adapter: MarketAdapter
    policy: Policy
    symbols: list[str]


async def run_all_markets(
    runs: list[MarketRun],
    logger: JsonlLogger,
    asof_day: date | None = None,
) -> dict[str, OrderResult | Exception]:
    """모든 시장의 일일 스텝을 동시 실행. 시장별 성공(OrderResult)/실패(Exception) 반환.

    실패한 시장은 daily_step_error 이벤트로 로그에 남는다 — 무인 운용 시 사후 감사용.
    """

    async def _one(run: MarketRun) -> OrderResult:
        return await run_daily_step(run.adapter, run.policy, run.symbols, logger, asof_day)

    outcomes = await asyncio.gather(*(_one(r) for r in runs), return_exceptions=True)

    results: dict[str, OrderResult | Exception] = {}
    for run, outcome in zip(runs, outcomes):
        market = run.adapter.market
        if isinstance(outcome, BaseException):
            logger.log(
                market,
                "daily_step_error",
                {
                    "policy": run.policy.name,
                    "error_type": type(outcome).__name__,
                    "error": str(outcome),
                },
            )
            results[market] = outcome if isinstance(outcome, Exception) else Exception(str(outcome))
        else:
            results[market] = outcome
    return results
