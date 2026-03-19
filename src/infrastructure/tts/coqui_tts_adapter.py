"""Coqui TTS primary adapter implementing TTSPort.

Synthesizes text to raw PCM audio chunks using the Coqui TTS library.
Model is configurable via the ``TTS_COQUI_MODEL`` environment variable.
Each yielded chunk is ≤ 4 KB (FR-016).
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
from typing import TYPE_CHECKING

import structlog

from src.domain.errors import TTSSynthesisError
from src.infrastructure.observability.circuit_breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_COQUI_MODEL: str = os.environ.get(
    "TTS_COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"
)
_CHUNK_SIZE: int = 4096  # 4 KB max per chunk (FR-016)

# ── Lazy model singleton ───────────────────────────────────────────────────────

_tts_instance = None
_tts_lock = asyncio.Lock()


async def _get_tts():  # type: ignore[return]  # noqa: ANN202
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance
    async with _tts_lock:
        if _tts_instance is not None:
            return _tts_instance
        try:
            from TTS.api import TTS  # type: ignore[import-untyped]

            logger.info("loading_coqui_tts_model", model=_COQUI_MODEL)
            _tts_instance = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: TTS(model_name=_COQUI_MODEL, progress_bar=False),
            )
        except Exception as exc:
            raise TTSSynthesisError(
                f"Failed to load Coqui TTS model '{_COQUI_MODEL}': {exc}"
            ) from exc
    return _tts_instance


def _float32_to_pcm16(samples: list[float]) -> bytes:
    """Convert float32 audio samples in [-1.0, 1.0] to 16-bit PCM bytes."""
    clipped = [max(-1.0, min(1.0, s)) for s in samples]
    return struct.pack(f"<{len(clipped)}h", *(int(s * 32767) for s in clipped))


# ── Adapter ───────────────────────────────────────────────────────────────────


class CoquiTTSAdapter:
    """TTS adapter backed by Coqui TTS.

    Implements the :class:`~src.application.ports.tts_port.TTSPort` Protocol.
    Yields raw PCM audio bytes in chunks of at most 4 KB.
    """

    def __init__(self) -> None:
        self._breaker = CircuitBreaker(name="coqui_tts")

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Generate PCM audio for *text* and yield it in ≤4 KB chunks."""
        return self._synthesize(text, language=language)

    async def _synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        tts = await _get_tts()

        try:
            async with self._breaker:
                # tts.tts() returns a list of float32 audio samples
                kwargs = {}
                if language:
                    kwargs["language"] = language

                samples: list[float] = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: tts.tts(text=text, **kwargs)
                )
        except CircuitOpenError as exc:
            raise TTSSynthesisError(f"Coqui TTS circuit breaker open: {exc}") from exc
        except TTSSynthesisError:
            raise
        except Exception as exc:
            raise TTSSynthesisError(f"Coqui TTS synthesis failed: {exc}") from exc

        # Convert float32 to PCM16 bytes and yield in chunks
        pcm_bytes = _float32_to_pcm16(samples)
        buf = io.BytesIO(pcm_bytes)
        while True:
            chunk = buf.read(_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
