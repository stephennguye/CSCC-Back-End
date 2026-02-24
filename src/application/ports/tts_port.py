"""TTSPort — abstract interface for Text-to-Speech adapters.

Zero infrastructure imports — pure Python typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@runtime_checkable
class TTSPort(Protocol):
    """Async streaming PCM audio chunk generator interface."""

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio as raw PCM chunks.

        Args:
            text:     The text to synthesize.
            voice:    Adapter-specific voice identifier.  If *None* the
                      adapter uses its configured default.
            language: BCP-47 language tag hint (e.g. ``"vi"``).

        Yields:
            Raw audio bytes.  Each chunk MUST be ≤ 4 KB (FR-016).
        """
        ...  # pragma: no cover
