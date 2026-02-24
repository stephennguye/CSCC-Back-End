"""HuggingFace Inference Client LLM fallback adapter implementing LLMPort.

Targets ``mistralai/Mistral-7B-Instruct-v0.2`` by default (configurable via
``HUGGINGFACE_FALLBACK_MODEL``).  Uses the ``huggingface_hub`` InferenceClient
with asyncio timeout wrapping.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.errors import LLMFallbackError, LLMFallbackExhaustedError, LLMTimeoutError
from src.infrastructure.observability.circuit_breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_DEFAULT_MODEL: str = os.environ.get(
    "HUGGINGFACE_FALLBACK_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.2",
)
_API_TOKEN: str | None = os.environ.get("HUGGINGFACE_API_TOKEN")
_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "10.0"))


# ── Adapter ───────────────────────────────────────────────────────────────────


class HuggingFaceLLMAdapter:
    """Fallback LLM adapter backed by ``huggingface_hub`` InferenceClient.

    Implements the :class:`~src.application.ports.llm_port.LLMPort` Protocol.
    Uses the HuggingFace text-generation inference endpoint with streaming.
    """

    def __init__(
        self,
        api_token: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_token = api_token or _API_TOKEN
        self._default_model = default_model or _DEFAULT_MODEL
        self._timeout = timeout or _TIMEOUT_SECONDS
        self._client: Any | None = None
        self._breaker = CircuitBreaker(name="huggingface_llm")

    def _get_client(self) -> Any:  # noqa: ANN401
        if self._client is None:
            from huggingface_hub import InferenceClient  # type: ignore[import-untyped]

            self._client = InferenceClient(
                token=self._api_token,
            )
        return self._client

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert OpenAI-style message list to Mistral instruction format."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(f" {content} ")
        return "".join(parts)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response tokens via HuggingFace Inference API."""
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
        prompt = self._build_prompt(messages)

        try:
            async with self._breaker:
                async with asyncio.timeout(self._timeout):
                    # huggingface_hub InferenceClient.text_generation is sync;
                    # run it off-thread and iterate the generator
                    def _run_sync():  # type: ignore[return]  # noqa: ANN202
                        return list(
                            client.text_generation(
                                prompt,
                                model=chosen_model,
                                max_new_tokens=max_tokens,
                                temperature=max(temperature, 0.01),
                                stream=True,
                                details=False,
                            )
                        )

                    tokens = await asyncio.get_event_loop().run_in_executor(None, _run_sync)
                    for token in tokens:
                        yield token
        except CircuitOpenError as exc:
            raise LLMFallbackExhaustedError(f"HuggingFace circuit breaker open: {exc}") from exc
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"HuggingFace call exceeded {self._timeout}s timeout"
            ) from exc
        except LLMTimeoutError:
            raise
        except Exception as exc:
            raise LLMFallbackExhaustedError(
                f"HuggingFace adapter error: {exc}"
            ) from exc

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
        # Extract JSON object from the raw response (model may add preamble)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("huggingface_json_parse_error", raw=raw[:200])
            raise LLMFallbackError(
                "HuggingFace response did not contain a JSON object"
            )
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError as exc:
            raise LLMFallbackError(
                f"HuggingFace returned invalid JSON: {exc}"
            ) from exc
