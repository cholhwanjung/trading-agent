"""어댑터 공통 재시도 유틸 — 네트워크 호출(브로커 API)의 일시 장애 흡수.

지수 백오프. 어댑터 구현체(ccxt/KIS/Alpaca)가 자기 호출을 감싸는 용도이며,
재시도로도 실패하면 예외를 그대로 올린다 — 시장별 격리는 harness.runner 가 담당.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """fn 을 최대 attempts 회 시도. 대기 시간은 base_delay * 2^(시도-1).

    마지막 시도의 예외는 감싸지 않고 그대로 전파한다(원인 보존).
    """

    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except exceptions:
            if attempt == attempts:
                raise
            await asyncio.sleep(base_delay * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # attempts >= 1 보장
