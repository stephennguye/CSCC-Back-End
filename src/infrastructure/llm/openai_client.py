"""OpenAI LLM adapter implementing LLMPort.

Streams tokens via AsyncOpenAI.  Supports structured JSON output mode,
configurable model override, and asyncio timeout wrapping.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.errors import LLMFallbackError, LLMTimeoutError
from src.infrastructure.observability.circuit_breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_DEFAULT_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "10.0"))


# ── Adapter ───────────────────────────────────────────────────────────────────


class OpenAILLMAdapter:
    """LLM adapter backed by AsyncOpenAI streaming API.

    Implements the :class:`~src.application.ports.llm_port.LLMPort` Protocol.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or _API_KEY
        self._default_model = default_model or _DEFAULT_MODEL
        self._timeout = timeout or _TIMEOUT_SECONDS
        self._client: Any | None = None
        self._breaker = CircuitBreaker(name="openai_llm")

    def _get_client(self) -> Any:  # noqa: ANN401
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response tokens via OpenAI Chat Completions API."""
        return self._stream_tokens(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    async def _stream_tokens(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()
        chosen_model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            async with self._breaker:
                async with asyncio.timeout(self._timeout):
                    stream = await client.chat.completions.create(**kwargs)
                    async for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content
        except CircuitOpenError as exc:
            raise LLMFallbackError(f"OpenAI circuit breaker open: {exc}") from exc
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"OpenAI call exceeded {self._timeout}s timeout"
            ) from exc
        except (LLMTimeoutError, LLMFallbackError):
            raise
        except Exception as exc:
            raise LLMFallbackError(f"OpenAI adapter error: {exc}") from exc

    async def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Collect streaming tokens and parse JSON response."""
        tokens: list[str] = []
        async for token in await self.generate_stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        ):
            tokens.append(token)
        raw = "".join(tokens)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("openai_json_parse_error", raw=raw[:200])
            raise LLMFallbackError(f"OpenAI returned invalid JSON: {exc}") from exc
