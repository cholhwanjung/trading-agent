"""LLM 백본 추상화 계약 ([ADR-008] · [ADR-014]).

Trader/Alpha Lab/Reflection 은 이 계약만 본다 — 프로바이더 교체는 env 설정만으로.
이중 속도(smart=결정·reflection / fast=요약·정리)는 tier 이름으로 노출된다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """프로바이더 응답 오류 (인증·형식·한도)."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMBackend(ABC):
    provider: str

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],  # [{"role": "user"|"assistant", "content": ...}]
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        json_mode: bool = False,  # 지원 프로바이더에서만 강제, 그 외 무시
    ) -> LLMResponse: ...

    @abstractmethod
    async def close(self) -> None: ...
