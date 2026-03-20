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

_MODEL_SIZE: str = os.environ.get("FASTER_WHISPER_MODEL", "small")
_DEVICE: str = os.environ.get("FASTER_WHISPER_DEVICE", "cpu")
_COMPUTE_TYPE: str = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")
_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ASR_CONFIDENCE_THRESHOLD", "0.6")
)
_DEFAULT_LANGUAGE: str = os.environ.get("FASTER_WHISPER_LANGUAGE", "vi")
_MIN_AUDIO_DURATION_SEC: float = float(
    os.environ.get("ASR_MIN_AUDIO_DURATION", "0.5")
)

# Prompt to bias Whisper toward Vietnamese airline booking vocabulary.
# This dramatically reduces hallucinations for domain-specific terms.
_INITIAL_PROMPT: str = (
    "Đặt vé máy bay, chuyến bay, hãng hàng không Vietnam Airlines, "
    "Vietjet Air, Bamboo Airways, sân bay Hà Nội, Đà Nẵng, Hồ Chí Minh, "
    "Sài Gòn, hạng phổ thông, hạng thương gia, hạng nhất, khứ hồi, "
    "một chiều, ngày bay, giờ bay, xác nhận, vâng, đúng rồi, có, không."
)

# Known Whisper hallucination patterns — these are injected when the model
# has very short or silent audio and produces nonsensical output.
_HALLUCINATION_PATTERNS: list[str] = [
    "subscribe",
    "like and share",
    "la la school",
    "theo dõi",
    "kênh",
    "video hấp dẫn",
    "bỏ lỡ",
    "hấp dẫn nhé",
    "các bạn",
    "cập nhật",
    "nhấn nút",
    "ghiền mì gõ",
    "hôm nay",
    "mọi người",
]

# Very short generic Vietnamese phrases Whisper produces from noise.
# These are only filtered when confidence is below the threshold AND
# audio duration is short, to avoid filtering real speech.
_SHORT_NOISE_PHRASES: set[str] = {
    "cảm ơn.",
    "vâng.",
    "chào.",
    "tạm biệt.",
    "hẹn gặp lại.",
    "xin chào.",
    "cảm ơn các bạn.",
    "cảm ơn mọi người.",
}

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


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> io.BytesIO:
    """Wrap raw PCM16 signed-integer audio bytes in a minimal WAV container.

    faster-whisper expects a file-like object with a WAV/FLAC/MP3 header.
    The frontend sends raw 16-bit signed PCM at 16 kHz mono.
    """
    import struct

    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    file_size = 36 + data_size  # 44 byte header - 8 byte RIFF preamble

    header = struct.pack(
        "<4sI4s"     # RIFF header
        "4sIHHIIHH"  # fmt chunk
        "4sI",       # data chunk header
        b"RIFF", file_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )

    buf = io.BytesIO()
    buf.write(header)
    buf.write(pcm_bytes)
    buf.seek(0)
    return buf


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
        # Default to Vietnamese — the primary language for this call center
        return self._transcribe(audio_chunks, language=language or _DEFAULT_LANGUAGE)

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
            logger.info("stt_empty_audio_buffer")
            return

        # Log audio stats for debugging
        num_samples = len(buffer) // 2  # 2 bytes per int16 sample
        duration_sec = num_samples / 16000
        logger.info(
            "stt_processing",
            audio_bytes=len(buffer),
            samples=num_samples,
            duration_sec=round(duration_sec, 2),
        )

        # Reject audio that is too short — likely noise triggering VAD
        if duration_sec < _MIN_AUDIO_DURATION_SEC:
            logger.info(
                "stt_audio_too_short",
                duration_sec=round(duration_sec, 2),
                min_duration=_MIN_AUDIO_DURATION_SEC,
            )
            return

        model = await _get_model()

        # Wrap raw PCM16 mono 16kHz bytes in a WAV header so faster-whisper
        # can decode them (it expects a file-like object with a valid header).
        audio_file = _pcm16_to_wav(bytes(buffer))

        try:
            async with self._breaker:
                segments, _info = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.transcribe(
                        audio_file,
                        language=language,
                        word_timestamps=False,
                        beam_size=5,
                        no_speech_threshold=0.4,
                        log_prob_threshold=-0.5,
                        condition_on_previous_text=False,
                        initial_prompt=_INITIAL_PROMPT,
                    ),
                )
        except CircuitOpenError as exc:
            raise TranscriptionError(f"STT circuit breaker open: {exc}") from exc
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"faster-whisper transcription failed: {exc}") from exc

        segment_count = 0
        for segment in segments:
            text = segment.text.strip()
            segment_id = str(uuid.uuid4())

            # avg_logprob is in range (-inf, 0]; convert to 0-1 confidence.
            logprob = segment.avg_logprob
            raw_confidence = max(0.0, min(1.0, 1.0 + logprob / 2.0))

            # Secondary filter: high no_speech_prob → likely not real speech.
            # For very high no_speech_prob (>0.9), always penalize.
            # For moderately high (>0.7), only penalize longer audio (>2s)
            # since short real utterances like "vâng" can have elevated no_speech_prob.
            if hasattr(segment, "no_speech_prob"):
                if segment.no_speech_prob > 0.9 or (segment.no_speech_prob > 0.7 and duration_sec > 2.0):
                    raw_confidence = min(raw_confidence, 0.1)

            # Hallucination filter: reject known garbage patterns
            text_lower = text.lower()
            is_hallucination = any(p in text_lower for p in _HALLUCINATION_PATTERNS)
            if is_hallucination:
                logger.warning(
                    "stt_hallucination_filtered",
                    text=text[:80],
                    confidence=round(raw_confidence, 2),
                )
                continue

            # Confidence gate: reject low-confidence segments.
            # For short audio (< 2s), also filter common short phrases
            # that Whisper hallucinates from background noise.
            if raw_confidence < _CONFIDENCE_THRESHOLD:
                if duration_sec < 2.0 or text_lower.strip().rstrip(".!?") in {
                    p.rstrip(".") for p in _SHORT_NOISE_PHRASES
                }:
                    logger.warning(
                        "stt_low_confidence_filtered",
                        text=text[:80],
                        confidence=round(raw_confidence, 2),
                        duration_sec=round(duration_sec, 2),
                    )
                    continue

            segment_count += 1

            logger.info(
                "stt_segment",
                text=text[:80],
                confidence=round(raw_confidence, 2),
                segment_id=segment_id,
            )

            yield TranscriptionChunk(
                text=text,
                confidence=raw_confidence,
                segment_id=segment_id,
                is_final=False,
            )

            yield TranscriptionChunk(
                text=text,
                confidence=raw_confidence,
                segment_id=segment_id,
                is_final=True,
            )

        logger.info(
            "stt_transcription_complete",
            segment_count=segment_count,
            audio_bytes=len(buffer),
        )
