"""프로바이더 카탈로그 + 이중 속도 라우터.

모델 지정은 `"provider:model"` 스펙 문자열 하나 — env 로 tier 를 바꾼다:
    LLM_SMART=anthropic:claude-sonnet-5      # 결정·reflection (고성능)
    LLM_FAST=anthropic:claude-haiku-4-5-20251001  # 요약·데이터 정리 (경량)
    LLM_SMART=openai:gpt-5.2 / gemini:gemini-3-pro / ollama:llama4 ... 자유 교체
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from llm.backends import AnthropicBackend, OpenAICompatBackend
from llm.base import LLMBackend, LLMError, LLMResponse, Usage, UsageSink

# 프로바이더 카탈로그 — kind=openai 는 전부 OpenAICompatBackend 로 처리
PROVIDERS: dict[str, dict] = {
    "anthropic": {"kind": "anthropic", "key_env": "ANTHROPIC_API_KEY"},
    "openai": {"kind": "openai", "key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "gemini": {
        "kind": "openai",
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    },
    "deepseek": {"kind": "openai", "key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com/v1"},
    "openrouter": {"kind": "openai", "key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
    "xai": {"kind": "openai", "key_env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1"},
    # 로컬 — 키 불필요, base_url 은 OLLAMA_BASE_URL 로 재정의 가능
    "ollama": {"kind": "openai", "key_env": None, "base_url": "http://localhost:11434/v1", "base_url_env": "OLLAMA_BASE_URL"},
}

DEFAULT_SMART = "anthropic:claude-sonnet-5"
DEFAULT_FAST = "anthropic:claude-haiku-4-5-20251001"
DEFAULT_EMBED = "openai:text-embedding-3-small"  # env LLM_EMBED 로 재정의

Tier = Literal["smart", "fast"]


def parse_spec(spec: str) -> tuple[str, str]:
    """`"provider:model"` → (provider, model). 모델명에 ':' 가 있어도 안전(첫 ':' 분리)."""
    provider, sep, model = spec.partition(":")
    if not sep or not model or provider not in PROVIDERS:
        raise LLMError(
            f"잘못된 LLM 스펙 {spec!r} — 'provider:model', provider ∈ {sorted(PROVIDERS)}"
        )
    return provider, model


def make_backend(provider: str, env: Mapping[str, str]) -> LLMBackend:
    cfg = PROVIDERS[provider]
    key = env.get(cfg["key_env"]) if cfg["key_env"] else None
    if cfg["key_env"] and not key:
        raise LLMError(f"provider={provider} 키 미설정 — .env 에 {cfg['key_env']} 필요")
    if cfg["kind"] == "anthropic":
        return AnthropicBackend(api_key=key)
    base_url = env.get(cfg.get("base_url_env", ""), "") or cfg["base_url"]
    return OpenAICompatBackend(provider=provider, base_url=base_url, api_key=key)


class LLMRouter:
    """tier(smart/fast) → (백엔드, 모델) 라우팅. 백엔드는 프로바이더별 1회 생성."""

    def __init__(
        self,
        env: Mapping[str, str],
        smart: str | None = None,
        fast: str | None = None,
        usage_sink: UsageSink | None = None,
    ) -> None:
        self._env = env
        self._specs: dict[Tier, tuple[str, str]] = {
            "smart": parse_spec(smart or env.get("LLM_SMART") or DEFAULT_SMART),
            "fast": parse_spec(fast or env.get("LLM_FAST") or DEFAULT_FAST),
        }
        self._backends: dict[str, LLMBackend] = {}
        self._usage_sink = usage_sink  # None = 사용량 미기록 (테스트·경량 경로)

    def spec(self, tier: Tier) -> tuple[str, str]:
        return self._specs[tier]

    def _backend(self, provider: str) -> LLMBackend:
        if provider not in self._backends:
            self._backends[provider] = make_backend(provider, self._env)
        return self._backends[provider]

    async def complete(
        self, tier: Tier, *, purpose: str = "", market: str = "", **kwargs
    ) -> LLMResponse:
        """purpose·market 는 사용량 귀속 라벨 — 백엔드로 넘기지 않고 싱크에만 기록한다."""
        provider, model = self._specs[tier]
        resp = await self._backend(provider).complete(model=model, **kwargs)
        if self._usage_sink is not None:
            self._usage_sink(
                Usage(
                    provider=resp.provider,
                    model=resp.model,
                    input_tokens=resp.input_tokens or 0,
                    output_tokens=resp.output_tokens or 0,
                    tier=tier,
                    purpose=purpose,
                    market=market,
                )
            )
        return resp

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """임베딩 — LLM_EMBED 스펙(기본 openai:text-embedding-3-small) 사용."""
        provider, model = parse_spec(self._env.get("LLM_EMBED") or DEFAULT_EMBED)
        backend = self._backend(provider)
        if not hasattr(backend, "embed"):
            raise LLMError(f"provider={provider} 는 임베딩 미지원 — LLM_EMBED 를 변경하라")
        return await backend.embed(model, texts)

    async def close(self) -> None:
        for backend in self._backends.values():
            await backend.close()
        self._backends.clear()
