"""Google Cloud Speech-to-Text adapter implementing STTPort.

Uses the REST API with an API key for simplicity — no service account needed.
Collects all audio, sends a single synchronous recognize request, and yields
TranscriptionChunk objects matching the STTPort protocol.
"""

from __future__ import annotations

import base64
import os
import uuid
from typing import TYPE_CHECKING

import httpx
import structlog

from src.application.ports.stt_port import TranscriptionChunk
from src.domain.errors import TranscriptionError
from src.infrastructure.observability.circuit_breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_API_KEY: str = os.environ.get("GOOGLE_CLOUD_API_KEY", "")
_DEFAULT_LANGUAGE: str = os.environ.get("GOOGLE_STT_LANGUAGE", "vi-VN")
_MODEL: str = os.environ.get("GOOGLE_STT_MODEL", "latest_long")
_SAMPLE_RATE: int = int(os.environ.get("GOOGLE_STT_SAMPLE_RATE", "16000"))
_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ASR_CONFIDENCE_THRESHOLD", "0.6")
)
_MIN_AUDIO_DURATION_SEC: float = float(
    os.environ.get("ASR_MIN_AUDIO_DURATION", "0.5")
)

_RECOGNIZE_URL = "https://speech.googleapis.com/v1/speech:recognize"
_MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB — Google Sync API limit


class GoogleCloudSTTAdapter:
    """STT adapter backed by Google Cloud Speech-to-Text REST API.

    Implements the :class:`~src.application.ports.stt_port.STTPort` Protocol.
    """

    def __init__(self) -> None:
        self._breaker = CircuitBreaker(name="google_cloud_stt")
        self._client = httpx.AsyncClient(timeout=30.0)
        if not _API_KEY:
            logger.warning("google_cloud_stt_no_api_key")

    async def close(self) -> None:
        await self._client.aclose()

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        language: str | None = None,
    ) -> AsyncGenerator[TranscriptionChunk, None]:
        """Collect audio, call Google Cloud STT, yield TranscriptionChunks."""
        async for chunk in self._transcribe(audio_chunks, language=language or _DEFAULT_LANGUAGE):
            yield chunk

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
            if len(buffer) > _MAX_AUDIO_BYTES:
                raise TranscriptionError("Audio exceeds maximum size for synchronous STT (10 MB)")

        if not buffer:
            logger.info("stt_empty_audio_buffer")
            return

        # Log audio stats
        num_samples = len(buffer) // 2  # 2 bytes per int16 sample
        duration_sec = num_samples / _SAMPLE_RATE
        logger.info(
            "stt_processing",
            audio_bytes=len(buffer),
            samples=num_samples,
            duration_sec=round(duration_sec, 2),
        )

        # Reject audio that is too short
        if duration_sec < _MIN_AUDIO_DURATION_SEC:
            logger.info(
                "stt_audio_too_short",
                duration_sec=round(duration_sec, 2),
                min_duration=_MIN_AUDIO_DURATION_SEC,
            )
            return

        # Encode audio as base64 for the REST API
        audio_b64 = base64.b64encode(bytes(buffer)).decode("ascii")

        request_body = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": _SAMPLE_RATE,
                "languageCode": language,
                "model": _MODEL,
                "enableAutomaticPunctuation": True,
                "enableWordConfidence": True,
            },
            "audio": {
                "content": audio_b64,
            },
        }

        try:
            async with self._breaker:
                response = await self._client.post(
                    _RECOGNIZE_URL,
                    params={"key": _API_KEY},
                    json=request_body,
                )

                if response.status_code != 200:
                    error_detail = response.text[:200]
                    logger.error(
                        "google_stt_api_error",
                        status=response.status_code,
                        detail=error_detail,
                    )
                    raise TranscriptionError(
                        f"Google Cloud STT API error {response.status_code}: {error_detail}"
                    )

                result = response.json()

        except CircuitOpenError as exc:
            raise TranscriptionError(f"STT circuit breaker open: {exc}") from exc
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Google Cloud STT request failed: {exc}") from exc

        # Parse response — results may be empty if no speech detected
        results = result.get("results", [])
        if not results:
            logger.info("stt_no_speech_detected")
            return

        segment_count = 0
        for res in results:
            alternatives = res.get("alternatives", [])
            if not alternatives:
                continue

            best = alternatives[0]
            text = best.get("transcript", "").strip()
            if not text:
                continue

            # Google returns confidence as 0.0-1.0 (only for final results)
            raw_confidence = best.get("confidence", 0.8)
            segment_id = str(uuid.uuid4())

            # Confidence gate
            if raw_confidence < _CONFIDENCE_THRESHOLD:
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
                is_final=True,
            )

        logger.info(
            "stt_transcription_complete",
            segment_count=segment_count,
            audio_bytes=len(buffer),
        )
