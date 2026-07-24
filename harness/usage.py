"""LLM 토큰 사용량 → 구조화 로그 싱크.

라우터가 모든 프로바이더 호출의 단일 초크포인트이므로, 여기서 usage 를 한 번만
기록하면 결정·debate·챗·alpha·reflection·self-improve·capability 경로가 전부 잡힌다.
호출부마다 계측을 배선할 필요가 없다.

토큰·모델(사실)만 남기고 달러 환산은 저장하지 않는다 — 단가표가 바뀌므로 비용은
읽는 쪽(대시보드)에서 파생한다. 로그는 시장별 디렉토리가 아니라 전용 `USAGE`
네임스페이스에 모은다(챗·reflection 등 시장 횡단 호출을 한 곳에서 집계).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from harness.jsonlog import JsonlLogger
from llm.base import Usage

USAGE_NS = "USAGE"  # JsonlLogger 네임스페이스 — 실제 귀속 시장은 for_market 필드로


def make_usage_sink(root: Path | str) -> Callable[[Usage], None]:
    """{root}/data/logs/USAGE/{날짜}.jsonl 에 llm_usage 이벤트를 append 하는 싱크."""

    logger = JsonlLogger(Path(root) / "data" / "logs")

    def sink(u: Usage) -> None:
        logger.log(
            USAGE_NS,
            "llm_usage",
            {
                "purpose": u.purpose,
                "for_market": u.market,
                "tier": u.tier,
                "provider": u.provider,
                "model": u.model,
                "in": u.input_tokens,
                "out": u.output_tokens,
            },
        )

    return sink
