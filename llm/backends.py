"""LLM 백엔드 2종 — Anthropic 네이티브 + OpenAI 호환.

OpenAI 호환 하나로 OpenAI · Gemini · DeepSeek · OpenRouter · xAI · Ollama 를 커버한다
(모두 /chat/completions 형식 제공). 프로바이더별 base_url 은 llm/router.py 카탈로그에.
"""

from __future__ import annotations

import httpx

from adapters.retry import with_retry
from llm.base import LLMBackend, LLMError, LLMResponse

TIMEOUT = 60.0
ANTHROPIC_VERSION = "2023-06-01"


def _check_status(resp: httpx.Response, provider: str) -> None:
    """429/5xx 는 재시도 가능(HTTPStatusError 전파), 그 외 4xx 는 즉시 LLMError.

    인증·형식 오류는 재시도해도 결과가 같다 — 백오프 낭비 없이 바로 실패.
    """
    if resp.status_code == 429 or resp.status_code >= 500:
        resp.raise_for_status()
    if resp.status_code >= 400:
        raise LLMError(f"provider={provider} http={resp.status_code} body={resp.text[:200]}")


class AnthropicBackend(LLMBackend):
    provider = "anthropic"

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com") -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            timeout=TIMEOUT,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        json_mode: bool = False,  # Anthropic 은 json 스위치가 없다 — 프롬프트로 지시
    ) -> LLMResponse:
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        async def call():
            resp = await self._client.post("/v1/messages", json=body)
            _check_status(resp, self.provider)
            return resp.json()

        try:
            data = await with_retry(call, exceptions=(httpx.HTTPError,))
        except httpx.HTTPStatusError as e:
            raise LLMError(f"provider=anthropic http={e.response.status_code} body={e.response.text[:200]}") from e

        text = "".join(part["text"] for part in data["content"] if part["type"] == "text")
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=data.get("model", model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


class OpenAICompatBackend(LLMBackend):
    def __init__(self, provider: str, base_url: str, api_key: str | None = None) -> None:
        self.provider = provider
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=TIMEOUT)

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        full_messages = ([{"role": "system", "content": system}] if system else []) + messages
        body: dict = {"model": model, "messages": full_messages}
        # OpenAI 최신(reasoning) 모델 호환: max_tokens 대신 max_completion_tokens,
        # temperature 는 기본값(1)만 허용 → 미전송. 타 호환사는 표준 파라미터.
        if self.provider == "openai":
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = temperature
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        async def call():
            resp = await self._client.post("/chat/completions", json=body)
            _check_status(resp, self.provider)
            return resp.json()

        try:
            data = await with_retry(call, exceptions=(httpx.HTTPError,))
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"provider={self.provider} http={e.response.status_code} body={e.response.text[:200]}"
            ) from e

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=data.get("model", model),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """OpenAI 호환 /embeddings — 메모리 relevancy·중복체크용 (R8)."""

        async def call():
            resp = await self._client.post("/embeddings", json={"model": model, "input": texts})
            _check_status(resp, self.provider)
            return resp.json()

        try:
            data = await with_retry(call, exceptions=(httpx.HTTPError,))
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"provider={self.provider} http={e.response.status_code} body={e.response.text[:200]}"
            ) from e
        return [item["embedding"] for item in data["data"]]
