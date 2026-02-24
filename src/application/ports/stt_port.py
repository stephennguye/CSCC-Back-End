"""STTPort — abstract interface for Speech-to-Text adapters.

Zero infrastructure imports — pure Python typing only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@dataclass
class TranscriptionChunk:
    """A single segment emitted by the STT pipeline."""

    text: str
    confidence: float  # 0.0 - 1.0
    segment_id: str
    is_final: bool = False


@runtime_checkable
class STTPort(Protocol):
    """Async streaming transcription interface."""

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        language: str | None = None,
    ) -> AsyncGenerator[TranscriptionChunk, None]:
        """Stream partial and final transcription segments.

        Args:
            audio_chunks: Async generator yielding raw PCM audio bytes.
            language:     BCP-47 language tag hint (e.g. ``"vi"`` for
                          Vietnamese).  If *None* the adapter auto-detects.

        Yields:
            :class:`TranscriptionChunk` objects.  Chunks with
            ``is_final=False`` are partial; ``is_final=True`` marks the
            end of an utterance segment.
        """
        ...  # pragma: no cover
