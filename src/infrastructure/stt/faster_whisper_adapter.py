"""faster-whisper STT adapter implementing STTPort.

Streams partial and final transcription segments from an audio byte stream.
Model size and device are configurable via environment variables.
"""

from __future__ import annotations

import asyncio
import io
import os
import uuid
from typing import TYPE_CHECKING

import structlog

from src.application.ports.stt_port import TranscriptionChunk
from src.domain.errors import TranscriptionError
from src.infrastructure.observability.circuit_breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_MODEL_SIZE: str = os.environ.get("FASTER_WHISPER_MODEL", "base")
_DEVICE: str = os.environ.get("FASTER_WHISPER_DEVICE", "cpu")
_COMPUTE_TYPE: str = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")
_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ASR_CONFIDENCE_THRESHOLD", "0.6")
)

# ── Lazy model singleton ───────────────────────────────────────────────────────

_model = None
_model_lock = asyncio.Lock()


async def _get_model():  # type: ignore[return]  # noqa: ANN202
    """Lazily load the faster-whisper model (once per process)."""
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            logger.info(
                "loading_faster_whisper_model",
                model_size=_MODEL_SIZE,
                device=_DEVICE,
                compute_type=_COMPUTE_TYPE,
            )
            _model = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: WhisperModel(
                    _MODEL_SIZE,
                    device=_DEVICE,
                    compute_type=_COMPUTE_TYPE,
                ),
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load faster-whisper model '{_MODEL_SIZE}': {exc}"
            ) from exc
    return _model


# ── Adapter ───────────────────────────────────────────────────────────────────


class FasterWhisperAdapter:
    """Streaming STT adapter backed by faster-whisper.

    Implements the :class:`~src.application.ports.stt_port.STTPort` Protocol.
    """

    def __init__(self) -> None:
        self._breaker = CircuitBreaker(name="faster_whisper_stt")

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        language: str | None = None,
    ) -> AsyncGenerator[TranscriptionChunk, None]:
        """Collect all audio bytes, run transcription off the event loop, and
        yield TranscriptionChunk objects (partial then final per segment).
        """
        return self._transcribe(audio_chunks, language=language)

    async def _transcribe(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        language: str | None = None,
    ) -> AsyncGenerator[TranscriptionChunk, None]:
        """Internal async generator for transcription."""
        # Collect all audio bytes from the stream
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)

        if not buffer:
            return

        model = await _get_model()

        # Run inference in thread-pool to avoid blocking the event loop
        audio_file = io.BytesIO(bytes(buffer))

        try:
            async with self._breaker:
                segments, _info = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.transcribe(
                        audio_file,
                        language=language,
                        word_timestamps=False,
                        beam_size=5,
                    ),
                )
        except CircuitOpenError as exc:
            raise TranscriptionError(f"STT circuit breaker open: {exc}") from exc
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"faster-whisper transcription failed: {exc}") from exc

        for segment in segments:
            segment_id = str(uuid.uuid4())
            # avg_logprob is in range (-inf, 0]; convert to 0-1 confidence
            raw_confidence = max(0.0, min(1.0, 1.0 + segment.avg_logprob))

            # Emit partial first
            yield TranscriptionChunk(
                text=segment.text.strip(),
                confidence=raw_confidence,
                segment_id=segment_id,
                is_final=False,
            )

            # Emit final
            yield TranscriptionChunk(
                text=segment.text.strip(),
                confidence=raw_confidence,
                segment_id=segment_id,
                is_final=True,
            )

        logger.debug("transcription_complete", chunks_emitted=True)
