"""gTTS adapter implementing TTSPort.

Uses Google Translate TTS (``gTTS`` Python package) to generate audio.
Supports Vietnamese via ``language="vi"``.  Yields chunks ≤ 4 KB (FR-016).

Note: gTTS is synchronous and produces MP3 output.  The adapter runs
synthesis in a thread executor and streams the resulting bytes.
"""

from __future__ import annotations

import asyncio
import io
import os
from typing import TYPE_CHECKING

import structlog

from src.domain.errors import TTSSynthesisError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_DEFAULT_LANGUAGE: str = os.environ.get("TTS_GTTS_LANGUAGE", "vi")
_CHUNK_SIZE: int = 4096  # 4 KB max per chunk (FR-016)


# ── Adapter ───────────────────────────────────────────────────────────────────


class GTTSAdapter:
    """TTS adapter backed by gTTS (Google Translate TTS).

    Implements the :class:`~src.application.ports.tts_port.TTSPort` Protocol.
    Yields raw audio bytes (MP3) in chunks of at most 4 KB.
    """

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio for *text* via gTTS and yield ≤4 KB chunks."""
        return self._synthesize(text, language=language)

    async def _synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        try:
            from gtts import gTTS  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TTSSynthesisError(
                "gTTS package is not installed; install it with `pip install gTTS`"
            ) from exc

        lang = language or _DEFAULT_LANGUAGE

        try:
            loop = asyncio.get_event_loop()
            buffer = io.BytesIO()

            def _synthesize_sync() -> None:
                tts = gTTS(text=text, lang=lang)
                tts.write_to_fp(buffer)

            await loop.run_in_executor(None, _synthesize_sync)

        except Exception as exc:
            raise TTSSynthesisError(f"gTTS synthesis failed: {exc}") from exc

        # Yield collected audio data in ≤4 KB chunks
        buffer.seek(0)
        while True:
            chunk = buffer.read(_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
