"""LLM 백본 추상화 계약.

Trader/Alpha Lab/Reflection 은 이 계약만 본다 — 프로바이더 교체는 env 설정만으로.
이중 속도(smart=결정·reflection / fast=요약·정리)는 tier 이름으로 노출된다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
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


@dataclass(frozen=True)
class Usage:
    """LLM 호출 1건의 토큰 사용량 + 귀속 라벨.

    비용(통화)은 저장하지 않는다 — 프로바이더 단가는 바뀌므로 토큰·모델만 사실로
    남기고, 달러 환산은 단가표를 가진 읽는 쪽(대시보드)에서 파생한다. purpose 는
    호출 목적(decision·debate·chat·alpha·reflection 등), market 은 귀속 시장(있으면).
    """

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    tier: str
    purpose: str
    market: str  # 시장 귀속이 있으면 그 시장, 시장 횡단 호출은 ""


# 라우터가 호출 1건마다 Usage 를 넘기는 싱크. 기록 방식(로그 파일 등)은 주입자 책임.
UsageSink = Callable[[Usage], None]


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
