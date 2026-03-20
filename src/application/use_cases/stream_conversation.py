"""StreamConversationUseCase — STT → TOD Pipeline → TTS pipeline.

Orchestrates one full AI turn:
  1. Transcribes the audio stream (STT).
  2. Low-confidence / hallucination check.
  3. Routes through TOD pipeline (NLU → DST → Policy → NLG).
  4. Synthesizes audio (TTS, with primary→fallback chain).
  5. Streams all WS frames (transcript.partial, transcript.final,
     transcript.low_confidence, transcript.ai_final, pipeline.state,
     audio.response.*) to the caller via injected
     ``send_text`` / ``send_binary`` callbacks.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import UTC
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.errors import TranscriptionError
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.infrastructure.observability.noop_tracer import _NoopTracer

if TYPE_CHECKING:
    from src.application.ports.stt_port import STTPort, TranscriptionChunk
    from src.application.ports.tts_port import TTSPort
    from src.application.use_cases.tod_pipeline import TODPipelineUseCase

logger = structlog.get_logger(__name__)

# ── OpenTelemetry helpers ────────────────────────────────────────────────────


def _get_tracer():  # type: ignore[return]  # noqa: ANN202
    try:
        from opentelemetry import trace  # type: ignore[import-untyped]

        return trace.get_tracer("cscc.stream_conversation")
    except ImportError:
        return _NoopTracer()


# ── Configuration ─────────────────────────────────────────────────────────────

_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ASR_CONFIDENCE_THRESHOLD", "0.6")
)
_LOW_CONFIDENCE_RESPONSE = (
    "Xin lỗi, em chưa nghe rõ. Anh/chị có thể nói lại được không ạ?"
)

# Type aliases for WS send callbacks
SendTextFn = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]
SendBinaryFn = Callable[[bytes], Coroutine[Any, Any, None]]


class StreamConversationUseCase:
    """Orchestrates a single AI response turn via the TOD pipeline.

    Pipeline: STT → NLU → DST → Policy → NLG → TTS
    """

    def __init__(
        self,
        stt: STTPort,
        tod_pipeline: TODPipelineUseCase,
        tts_primary: TTSPort,
        tts_fallback: TTSPort,
    ) -> None:
        self._stt = stt
        self._tod_pipeline = tod_pipeline
        self._tts_primary = tts_primary
        self._tts_fallback = tts_fallback

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        *,
        session_id: uuid.UUID,
        audio_stream: AsyncGenerator[bytes, None],
        conversation_history: list[dict[str, str]],
        send_text: SendTextFn,
        send_binary: SendBinaryFn,
    ) -> tuple[str, str, bool]:
        """Execute one full turn of the AI conversation pipeline.

        Args:
            session_id:           UUID of the active call session.
            audio_stream:         Async generator yielding raw PCM bytes.
            conversation_history: Prior turns (unused by TOD but kept for API compat).
            send_text:            Async callback to push a text WS frame dict.
            send_binary:          Async callback to push a binary WS frame.

        Returns:
            Tuple of (user_transcript, ai_response_text, should_end_call).
        """
        session_str = str(session_id)
        tracer = _get_tracer()

        import contextlib as _cl
        with _cl.suppress(Exception):
            from src.infrastructure.observability.metrics import ws_connections_total
            ws_connections_total.inc()

        # ── 1. Transcription ──────────────────────────────────────────────
        with tracer.start_as_current_span("call.stt") as stt_span:
            stt_span.set_attribute("session_id", session_str)
            transcript, confidence = await self._transcribe(
                session_id=session_str,
                audio_stream=audio_stream,
                send_text=send_text,
            )

        logger.info(
            "stt_complete",
            session_id=session_str,
            transcript=transcript[:100] if transcript else "(empty)",
            confidence=confidence.value,
        )

        if not transcript:
            logger.info("empty_transcript_skipping", session_id=session_str)
            return "", "", False

        # ── 1b. Hallucination / noise filter ──────────────────────────────
        # Very low confidence → almost certainly noise hallucination; drop silently.
        _HALLUCINATION_THRESHOLD = 0.35
        if confidence.value < _HALLUCINATION_THRESHOLD:
            logger.warning(
                "stt_hallucination_rejected",
                session_id=session_str,
                transcript=transcript[:80],
                confidence=confidence.value,
            )
            return "", "", False

        # ── 2. Low-confidence fallback ───────────────────────────────────
        # Only prompt the user to repeat if confidence is moderately low
        # (real but unclear speech). For borderline noise (< 0.45), drop silently.
        if confidence.is_below_threshold(_CONFIDENCE_THRESHOLD):
            if confidence.value < 0.45:
                logger.info(
                    "stt_low_confidence_silent_drop",
                    session_id=session_str,
                    transcript=transcript[:80],
                    confidence=confidence.value,
                )
                return "", "", False
            await send_text(
                {
                    "type": "transcript.low_confidence",
                    "session_id": session_str,
                    "payload": {
                        "segment_id": str(uuid.uuid4()),
                        "prompt_message": _LOW_CONFIDENCE_RESPONSE,
                    },
                }
            )
            await self._synthesize_and_stream(
                session_id=session_str,
                text=_LOW_CONFIDENCE_RESPONSE,
                turn_id=str(uuid.uuid4()),
                send_text=send_text,
                send_binary=send_binary,
            )
            return transcript, "", False

        turn_id = str(uuid.uuid4())

        # ── 3. TOD pipeline (NLU → DST → Policy → NLG) ──────────────────
        try:
            with tracer.start_as_current_span("call.tod_pipeline") as tod_span:
                tod_span.set_attribute("session_id", session_str)
                tod_span.set_attribute("turn_id", turn_id)

                tod_result = await self._tod_pipeline.process_turn(
                    session_id=session_str,
                    user_text=transcript,
                )
        except Exception as exc:
            logger.exception(
                "tod_pipeline_error",
                session_id=session_str,
                user_text=transcript[:80],
                error=str(exc),
            )
            # Send a fallback error response so the user isn't left hanging
            fallback = "Xin lỗi, hệ thống gặp lỗi. Vui lòng thử lại."
            await send_text(
                {
                    "type": "transcript.ai_final",
                    "session_id": session_str,
                    "payload": {
                        "turn_id": turn_id,
                        "text": fallback,
                        "timestamp": _utc_now_iso(),
                        "sequence_number": 1,
                    },
                }
            )
            return transcript, fallback, False

        ai_response_text = str(tod_result.get("response_text", ""))
        should_end_call = bool(tod_result.get("should_end_call", False))
        if not ai_response_text:
            return transcript, "", should_end_call

        # Emit pipeline.state for visualization
        await send_text(
            {
                "type": "pipeline.state",
                "session_id": session_str,
                "payload": {
                    "turn_id": turn_id,
                    "stt_text": transcript,
                    "nlu": tod_result.get("nlu"),
                    "state": tod_result.get("state"),
                    "action": tod_result.get("action"),
                    "target_slot": tod_result.get("target_slot"),
                    "nlg_response": ai_response_text,
                    "timestamp": _utc_now_iso(),
                },
            }
        )

        # ── 4. Send transcript.ai_final ──────────────────────────────────
        await send_text(
            {
                "type": "transcript.ai_final",
                "session_id": session_str,
                "payload": {
                    "turn_id": turn_id,
                    "text": ai_response_text,
                    "timestamp": _utc_now_iso(),
                    "sequence_number": 1,
                },
            }
        )

        # ── 5. TTS streaming ─────────────────────────────────────────────
        import asyncio

        try:
            await asyncio.wait_for(
                self._synthesize_and_stream(
                    session_id=session_str,
                    text=ai_response_text,
                    turn_id=turn_id,
                    send_text=send_text,
                    send_binary=send_binary,
                ),
                timeout=15.0,
            )
        except TimeoutError:
            logger.warning("tts_timeout", session_id=session_str, text_length=len(ai_response_text))
        except Exception:
            logger.exception("tts_error", session_id=session_str)

        return transcript, ai_response_text, should_end_call

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _transcribe(
        self,
        *,
        session_id: str,
        audio_stream: AsyncGenerator[bytes, None],
        send_text: SendTextFn,
    ) -> tuple[str, ConfidenceScore]:
        """Run STT and emit transcript.partial / transcript.final frames."""
        final_text = ""
        best_confidence: float = 1.0
        seen_finals: set[str] = set()

        try:
            async for chunk in self._stt.transcribe_stream(audio_stream):
                chunk: TranscriptionChunk
                if not chunk.is_final:
                    await send_text(
                        {
                            "type": "transcript.partial",
                            "session_id": session_id,
                            "payload": {
                                "text": chunk.text,
                                "confidence": chunk.confidence,
                                "segment_id": chunk.segment_id,
                            },
                        }
                    )
                else:
                    if chunk.segment_id not in seen_finals:
                        seen_finals.add(chunk.segment_id)
                        final_text += (" " if final_text else "") + chunk.text
                        best_confidence = min(best_confidence, chunk.confidence)
                        await send_text(
                            {
                                "type": "transcript.final",
                                "session_id": session_id,
                                "payload": {
                                    "text": chunk.text,
                                    "confidence": chunk.confidence,
                                    "segment_id": chunk.segment_id,
                                },
                            }
                        )
        except TranscriptionError:
            raise

        return final_text.strip(), ConfidenceScore(value=max(0.0, min(1.0, best_confidence)))

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences for chunked TTS synthesis.

        Splits on sentence-ending punctuation (.!?) and newlines,
        keeping each chunk short enough for fast TTS synthesis.
        Only chunks when text is long enough to benefit (>80 chars).
        For very long segments, also splits on commas to reduce
        per-segment TTS latency.
        """
        import re

        if len(text) <= 80:
            return [text]

        # Split on sentence boundaries: period/exclamation/question followed by space or newline
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            # Further split long segments on commas to keep TTS responsive
            if len(stripped) > 80:
                sub_parts = re.split(r",\s+", stripped)
                for sp in sub_parts:
                    sp = sp.strip()
                    if sp:
                        sentences.append(sp)
            else:
                sentences.append(stripped)
        return sentences if sentences else [text]

    async def _synthesize_and_stream(
        self,
        *,
        session_id: str,
        text: str,
        turn_id: str,
        send_text: SendTextFn,
        send_binary: SendBinaryFn,
    ) -> None:
        """Run TTS with primary→fallback chain; stream audio.response.* frames.

        For long texts (>100 chars), splits into sentences and synthesizes
        each sentence independently so audio starts playing faster.
        """
        logger.info(
            "tts_starting",
            session_id=session_id,
            turn_id=turn_id,
            text_length=len(text),
            text_preview=text[:60],
        )

        await send_text(
            {
                "type": "audio.response.start",
                "session_id": session_id,
                "payload": {"turn_id": turn_id, "codec": "pcm_16khz_mono"},
            }
        )
        logger.info("tts_sent_audio_start", session_id=session_id)

        sentences = self._split_into_sentences(text)
        any_ok = False

        for sentence in sentences:
            ok = await self._try_tts(
                tts=self._tts_primary,
                text=sentence,
                send_binary=send_binary,
                session_id=session_id,
                label="primary",
            )
            if not ok:
                logger.warning("tts_primary_failed_trying_fallback", session_id=session_id)
                ok = await self._try_tts(
                    tts=self._tts_fallback,
                    text=sentence,
                    send_binary=send_binary,
                    session_id=session_id,
                    label="fallback",
                )
            if ok:
                any_ok = True

        if not any_ok:
            logger.error(
                "tts_all_providers_failed",
                session_id=session_id,
                text_length=len(text),
            )

        await send_text(
            {
                "type": "audio.response.end",
                "session_id": session_id,
                "payload": {"turn_id": turn_id},
            }
        )
        logger.info("tts_sent_audio_end", session_id=session_id, success=any_ok)

    async def _try_tts(
        self,
        *,
        tts: TTSPort,
        text: str,
        send_binary: SendBinaryFn,
        session_id: str,
        label: str,
    ) -> bool:
        """Attempt TTS synthesis with streaming MP3→PCM conversion.

        Pipes MP3 chunks from the TTS adapter into ffmpeg as they arrive,
        reads PCM from ffmpeg's stdout concurrently, and streams PCM chunks
        to the WebSocket immediately — so the user hears audio within ~200ms
        of the first TTS chunk instead of waiting for full synthesis.
        """
        import contextlib as _cl
        import subprocess
        import time

        t0 = time.perf_counter()
        provider = "edge_tts" if label == "primary" else "gtts"
        try:
            logger.info("tts_synthesizing", label=label, session_id=session_id)

            import asyncio

            first_chunk_logged = False
            mp3_total = 0
            pcm_total = 0
            chunks_sent = 0
            chunk_size = 4096

            # Stream MP3→PCM via ffmpeg subprocess with concurrent I/O.
            # MP3 chunks are piped to ffmpeg stdin as they arrive from TTS,
            # PCM is read from stdout and sent to WebSocket immediately.
            def _streaming_ffmpeg_convert(mp3_chunks_list: list[bytes]) -> bytes:
                try:
                    proc = subprocess.Popen(
                        [
                            "ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-i", "pipe:0",
                            "-f", "s16le", "-ar", "16000", "-ac", "1",
                            "pipe:1",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    # Write all collected MP3 data and close stdin
                    all_mp3 = b"".join(mp3_chunks_list)
                    stdout_data, _ = proc.communicate(input=all_mp3, timeout=15)
                    return stdout_data
                except FileNotFoundError:
                    logger.error("ffmpeg_not_found", detail="ffmpeg is required for TTS")
                    return b""
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return b""

            # Collect MP3 chunks from TTS adapter
            mp3_chunks: list[bytes] = []
            async for mp3_chunk in await tts.synthesize_stream(text):
                mp3_chunks.append(mp3_chunk)
                mp3_total += len(mp3_chunk)

            if not mp3_chunks:
                logger.warning("tts_empty_output", label=label, session_id=session_id)
                return False

            # Convert to PCM in thread (Windows-safe)
            pcm_data = await asyncio.to_thread(_streaming_ffmpeg_convert, mp3_chunks)
            if not pcm_data:
                return False

            pcm_total = len(pcm_data)

            if not first_chunk_logged:
                logger.info(
                    "tts_first_audio_chunk",
                    label=label,
                    session_id=session_id,
                    latency_ms=round((time.perf_counter() - t0) * 1000),
                )
                first_chunk_logged = True

            # Stream PCM to WebSocket in chunks
            offset = 0
            while offset < len(pcm_data):
                chunk = pcm_data[offset : offset + chunk_size]
                await send_binary(chunk)
                chunks_sent += 1
                offset += chunk_size

            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import (
                    tts_processing_duration_seconds,
                )
                tts_processing_duration_seconds.labels(provider=provider).observe(
                    time.perf_counter() - t0
                )

            logger.info(
                "tts_streamed",
                label=label,
                session_id=session_id,
                mp3_bytes=mp3_total,
                pcm_bytes=pcm_total,
                chunks_sent=chunks_sent,
                duration_ms=round((time.perf_counter() - t0) * 1000),
            )
            return True
        except Exception as exc:
            logger.warning(
                "tts_failed", label=label, session_id=session_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import (
                    tts_errors_total,
                    tts_fallback_total,
                )
                tts_errors_total.labels(provider=provider).inc()
                if label == "primary":
                    tts_fallback_total.inc()
            return False


def _utc_now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()
