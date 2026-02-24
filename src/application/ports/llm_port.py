"""LLMPort — abstract interface for Language Model adapters.

Application layer depends only on this Protocol; infrastructure adapters
implement it without the Application layer knowing anything about the
concrete SDK being used.

Zero infrastructure imports — pure Python typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@runtime_checkable
class LLMPort(Protocol):
    """Async streaming token generator interface."""

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM output token-by-token.

        Args:
            messages:    OpenAI-compatible message list
                         ``[{"role": "system"|"user"|"assistant", "content": "..."}]``.
            model:       Override the default model.  If *None* the adapter
                         uses its own configured default.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens:  Upper bound on generated tokens.
            json_mode:   When *True* the adapter instructs the model to emit
                         valid JSON only (structured output mode, FR-014).

        Yields:
            Individual string tokens as they arrive from the provider.
        """
        ...  # pragma: no cover

    async def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Convenience helper: collect the full streaming response and parse JSON.

        Implementations that do not support native JSON mode SHOULD set
        ``json_mode=True`` and collect tokens into a single string before
        calling ``json.loads``.
        """
        ...  # pragma: no cover
