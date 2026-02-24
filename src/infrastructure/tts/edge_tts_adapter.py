"""edge-tts fallback TTS adapter implementing TTSPort.

Uses Microsoft Edge TTS (``edge-tts`` Python package) to generate PCM audio.
Supports Vietnamese voice selection.  Yields chunks ≤ 4 KB (FR-016).
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

import structlog

from src.domain.errors import LLMFallbackError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_DEFAULT_VOICE: str = os.environ.get("TTS_EDGE_VOICE", "vi-VN-HoaiMyNeural")
_CHUNK_SIZE: int = 4096  # 4 KB max per chunk (FR-016)


# ── Adapter ───────────────────────────────────────────────────────────────────


class EdgeTTSAdapter:
    """Fallback TTS adapter backed by edge-tts.

    Implements the :class:`~src.application.ports.tts_port.TTSPort` Protocol.
    Yields raw audio bytes in chunks of at most 4 KB.

    Note: edge-tts natively outputs MP3/audio data.  For PCM, we rely on
    the caller receiving raw bytes that can be piped through a decoder.
    When used as a binary streaming fallback over WebSocket, the bytes are
    forwarded as-is; PCM conversion would require ``ffmpeg`` or similar.
    """

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio for *text* via edge-tts and yield ≤4 KB chunks."""
        return self._synthesize(text, voice=voice)

    async def _synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        try:
            import edge_tts  # type: ignore[import-untyped]
        except ImportError as exc:
            raise LLMFallbackError(
                "edge-tts package is not installed; install it with `pip install edge-tts`"
            ) from exc

        chosen_voice = voice or _DEFAULT_VOICE

        try:
            communicate = edge_tts.Communicate(text, voice=chosen_voice)
            buffer = io.BytesIO()

            async for chunk_type, chunk_data in communicate.stream():
                if chunk_type == "audio":
                    buffer.write(chunk_data)

        except Exception as exc:
            raise LLMFallbackError(f"edge-tts synthesis failed: {exc}") from exc

        # Yield collected audio data in ≤4 KB chunks
        buffer.seek(0)
        while True:
            chunk = buffer.read(_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
