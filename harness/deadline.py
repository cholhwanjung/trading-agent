"""런 전체 데드라인 워치독 — 어떤 스텝도 무한정 살지 않도록 총량 상한.

per-call timeout·retry 는 개별 호출만 막는다(각각 유한하나 호출 수의 합에는 상한이
없다). 이 데드라인은 런 전체(netgate + 파이프라인) aggregate 상한이다. 초과 시
코루틴을 취소해 finally 정리(어댑터·라우터 close)를 태우고, OS 가 파일/SQLite 락을
해제하도록 프로세스가 확실히 끝나게 한다.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine
from typing import Any

# netgate(≤600s) + 파이프라인 여유. 정상 런은 수 분이므로 병리적 지연만 걸린다.
# 환경변수 RUN_DEADLINE_S 로 override (느린 LLM 프로바이더 대비 상향 등).
DEFAULT_DEADLINE_S = 1500.0


def run_deadline_s() -> float:
    """환경변수 RUN_DEADLINE_S(초) 또는 기본값. 파싱 실패 시 기본값."""

    try:
        return float(os.environ.get("RUN_DEADLINE_S", DEFAULT_DEADLINE_S))
    except ValueError:
        return DEFAULT_DEADLINE_S


async def with_deadline(
    coro: Coroutine[Any, Any, int], *, label: str, deadline_s: float | None = None
) -> int:
    """coro 를 데드라인 안에 완료. 초과 시 취소·구조화 로그·exit code 1 반환.

    coro 의 finally(자원 close)는 취소 전파 중 실행돼 락을 해제한다.
    deadline_s 미지정 시 run_deadline_s() 사용(프로덕션 기본).
    """

    deadline = deadline_s if deadline_s is not None else run_deadline_s()
    try:
        return await asyncio.wait_for(coro, timeout=deadline)
    except asyncio.TimeoutError:
        print(f"status=fail event=run_deadline_exceeded label={label} deadline_s={deadline:.0f}")
        return 1
