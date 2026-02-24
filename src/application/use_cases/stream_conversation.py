"""StreamConversationUseCase — STT → PromptSanitizer → RAG (stub) → LLM → TTS pipeline.

Orchestrates one full AI turn:
  1. Transcribes the audio stream (STT).
  2. Sanitizes user input (PromptSanitizer).
  3. Retrieves RAG context (stub — pass-through, populated in Phase 4 / T040).
  4. Generates an AI response (LLM, with primary→fallback chain).
  5. Synthesizes audio (TTS, with primary→fallback chain).
  6. Streams all WS frames (transcript.partial, transcript.final,
     transcript.low_confidence, response.token, audio.response.*) to the
     caller via injected ``send_text`` / ``send_binary`` callbacks.
  7. Monitors the Redis barge-in Pub/Sub channel and cancels in-flight
     LLM/TTS whenever the caller starts speaking during an AI response.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import UTC
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.errors import (
    LLMFallbackError,
    LLMFallbackExhaustedError,
    LLMTimeoutError,
    PromptInjectionDetectedError,
    TranscriptionError,
)
from src.domain.value_objects.confidence_score import ConfidenceScore

if TYPE_CHECKING:
    from src.application.ports.llm_port import LLMPort
    from src.application.ports.stt_port import STTPort, TranscriptionChunk
    from src.application.ports.tts_port import TTSPort
    from src.application.use_cases.retrieve_knowledge import RetrieveKnowledgeUseCase
    from src.domain.services.prompt_sanitizer import PromptSanitizer
    from src.infrastructure.cache.redis_client import RedisClient

logger = structlog.get_logger(__name__)

# ── OpenTelemetry helpers ────────────────────────────────────────────────────


def _get_tracer():  # type: ignore[return]  # noqa: ANN202
    try:
        from opentelemetry import trace  # type: ignore[import-untyped]

        return trace.get_tracer("cscc.stream_conversation")
    except ImportError:
        return _NoopTracer()


class _NoopSpan:
    def set_attribute(self, _key: str, _value: object) -> None:
        pass

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *_: object) -> None:
        pass


class _NoopTracer:
    def start_as_current_span(self, _name: str, **_kw: object) -> _NoopSpan:
        return _NoopSpan()


# ── Configuration ─────────────────────────────────────────────────────────────

_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ASR_CONFIDENCE_THRESHOLD", "0.6")
)
_LOW_CONFIDENCE_PROMPT = (
    "I'm sorry, I didn't quite catch that. Could you please repeat?"
)
_LLM_FALLBACK_TIMEOUT: float = 2.0  # seconds before switching to fallback

# Type aliases for WS send callbacks
SendTextFn = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]
SendBinaryFn = Callable[[bytes], Coroutine[Any, Any, None]]


class StreamConversationUseCase:
    """Orchestrates a single AI response turn.

    Injected via FastAPI DI by :func:`~src.main.create_app`.
    """

    def __init__(
        self,
        stt: STTPort,
        llm_primary: LLMPort,
        llm_fallback: LLMPort,
        tts_primary: TTSPort,
        tts_fallback: TTSPort,
        prompt_sanitizer: PromptSanitizer,
        redis: RedisClient,
        retrieve_knowledge: RetrieveKnowledgeUseCase | None = None,
    ) -> None:
        self._stt = stt
        self._llm_primary = llm_primary
        self._llm_fallback = llm_fallback
        self._tts_primary = tts_primary
        self._tts_fallback = tts_fallback
        self._sanitizer = prompt_sanitizer
        self._redis = redis
        self._retrieve_knowledge = retrieve_knowledge

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
    ) -> str:
        """Execute one full turn of the AI conversation pipeline.

        Args:
            session_id:           UUID of the active call session.
            audio_stream:         Async generator yielding raw PCM bytes.
            conversation_history: OpenAI-style message list for context.
            send_text:            Async callback to push a text WS frame dict.
            send_binary:          Async callback to push a binary WS frame.

        Returns:
            The full AI response text (collected from the LLM token stream).
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

        if not transcript:
            logger.debug("empty_transcript", session_id=session_str)
            return ""

        # ── 2. Low-confidence fallback ─────────────────────────────────────
        if confidence.is_below_threshold(_CONFIDENCE_THRESHOLD):
            await send_text(
                {
                    "type": "transcript.low_confidence",
                    "session_id": session_str,
                    "payload": {
                        "segment_id": str(uuid.uuid4()),
                        "prompt_message": _LOW_CONFIDENCE_PROMPT,
                    },
                }
            )
            await self._synthesize_and_stream(
                session_id=session_str,
                text=_LOW_CONFIDENCE_PROMPT,
                turn_id=str(uuid.uuid4()),
                send_text=send_text,
                send_binary=send_binary,
            )
            return ""

        # ── 3. Prompt sanitization ─────────────────────────────────────────
        try:
            safe_transcript = self._sanitizer.sanitize(transcript)
        except PromptInjectionDetectedError:
            logger.warning("prompt_injection_blocked", session_id=session_str)
            raise

        # ── 4. RAG context retrieval ───────────────────────────────────────
        rag_context: str = ""
        if self._retrieve_knowledge is not None:
            rag_context = await self._retrieve_knowledge.execute(safe_transcript)

        # ── 5. Build LLM messages ──────────────────────────────────────────
        messages = self._build_messages(
            conversation_history=conversation_history,
            user_transcript=safe_transcript,
            rag_context=rag_context,
        )

        # ── 6. LLM streaming + barge-in monitoring ─────────────────────────
        turn_id = str(uuid.uuid4())
        ai_response_tokens: list[str] = []

        with tracer.start_as_current_span("call.llm") as llm_span:
            llm_span.set_attribute("session_id", session_str)
            llm_span.set_attribute("turn_id", turn_id)

            llm_task = asyncio.create_task(
                self._stream_llm_tokens(
                    session_id=session_str,
                    messages=messages,
                    turn_id=turn_id,
                    send_text=send_text,
                    ai_tokens=ai_response_tokens,
                )
            )
            barge_in_task = asyncio.create_task(
                self._watch_barge_in(session_id=session_str)
            )

            done, pending = await asyncio.wait(
                {llm_task, barge_in_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if barge_in_task in done and not barge_in_task.cancelled():
            # Barge-in occurred — abort the pipeline
            logger.info("barge_in_detected", session_id=session_str)
            return ""

        # Re-raise any LLM exception
        if llm_task in done:
            exc = llm_task.exception()
            if exc is not None:
                raise exc

        ai_response_text = "".join(ai_response_tokens)
        if not ai_response_text:
            return ""

        # ── 7. TTS streaming ──────────────────────────────────────────────
        # Send transcript.ai_final before TTS starts
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

        with tracer.start_as_current_span("call.tts") as tts_span:
            tts_span.set_attribute("session_id", session_str)
            tts_span.set_attribute("turn_id", turn_id)
            await self._synthesize_and_stream(
                session_id=session_str,
                text=ai_response_text,
                turn_id=turn_id,
                send_text=send_text,
                send_binary=send_binary,
            )

        return ai_response_text

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
            async for chunk in await self._stt.transcribe_stream(audio_stream):
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

    async def _stream_llm_tokens(
        self,
        *,
        session_id: str,
        messages: list[dict[str, str]],
        turn_id: str,
        send_text: SendTextFn,
        ai_tokens: list[str],
    ) -> None:
        """Call LLM with primary→fallback chain; emit response.token per token."""
        import time

        t0 = time.perf_counter()
        provider = "openai"
        try:
            async with asyncio.timeout(_LLM_FALLBACK_TIMEOUT):
                async for token in await self._llm_primary.generate_stream(messages):
                    ai_tokens.append(token)
                    await send_text(
                        {
                            "type": "response.token",
                            "session_id": session_id,
                            "payload": {"token": token, "turn_id": turn_id},
                        }
                    )
            import contextlib as _cl
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import (
                    llm_request_duration_seconds,
                )
                llm_request_duration_seconds.labels(provider=provider).observe(
                    time.perf_counter() - t0
                )
            return
        except (LLMTimeoutError, LLMFallbackError, TimeoutError) as exc:
            logger.warning(
                "llm_primary_failed_switching_to_fallback",
                session_id=session_id,
                error=str(exc),
            )
            import contextlib as _cl
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import llm_fallback_total
                llm_fallback_total.labels(
                    from_provider="openai", to_provider="huggingface"
                ).inc()
            ai_tokens.clear()

        # Fallback
        provider = "huggingface"
        t0 = time.perf_counter()
        try:
            async for token in await self._llm_fallback.generate_stream(messages):
                ai_tokens.append(token)
                await send_text(
                    {
                        "type": "response.token",
                        "session_id": session_id,
                        "payload": {"token": token, "turn_id": turn_id},
                    }
                )
            import contextlib as _cl
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import (
                    llm_request_duration_seconds,
                )
                llm_request_duration_seconds.labels(provider=provider).observe(
                    time.perf_counter() - t0
                )
        except Exception as exc:
            import contextlib as _cl
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import llm_errors_total
                llm_errors_total.labels(
                    provider=provider, error_type="fallback_exhausted"
                ).inc()
            raise LLMFallbackExhaustedError(
                f"Both LLM providers failed: {exc}"
            ) from exc

    async def _watch_barge_in(self, *, session_id: str) -> None:
        """Wait for a barge-in signal on the Redis Pub/Sub channel.

        Returns normally when a ``cancel`` message is received;
        raises ``asyncio.CancelledError`` if the task is cancelled externally.
        """
        pubsub = await self._redis.subscribe_barge_in(session_id)
        try:
            async for message in pubsub.listen():
                if message.get("type") == "message":
                    return  # barge-in signal received
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()

    async def _synthesize_and_stream(
        self,
        *,
        session_id: str,
        text: str,
        turn_id: str,
        send_text: SendTextFn,
        send_binary: SendBinaryFn,
    ) -> None:
        """Run TTS with primary→fallback chain; stream audio.response.* frames."""
        await send_text(
            {
                "type": "audio.response.start",
                "session_id": session_id,
                "payload": {"turn_id": turn_id, "codec": "pcm_16khz_mono"},
            }
        )

        success = await self._try_tts(
            tts=self._tts_primary,
            text=text,
            send_binary=send_binary,
            session_id=session_id,
            label="primary",
        )
        if not success:
            await self._try_tts(
                tts=self._tts_fallback,
                text=text,
                send_binary=send_binary,
                session_id=session_id,
                label="fallback",
            )

        await send_text(
            {
                "type": "audio.response.end",
                "session_id": session_id,
                "payload": {"turn_id": turn_id},
            }
        )

    async def _try_tts(
        self,
        *,
        tts: TTSPort,
        text: str,
        send_binary: SendBinaryFn,
        session_id: str,
        label: str,
    ) -> bool:
        """Attempt to stream TTS audio; return True on success, False on failure."""
        import time

        t0 = time.perf_counter()
        provider = "coqui" if label == "primary" else "edge_tts"
        try:
            async for chunk in await tts.synthesize_stream(text):
                await send_binary(chunk)
            import contextlib as _cl
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import (
                    tts_processing_duration_seconds,
                )
                tts_processing_duration_seconds.labels(provider=provider).observe(
                    time.perf_counter() - t0
                )
            return True
        except Exception as exc:
            logger.warning(
                "tts_failed", label=label, session_id=session_id, error=str(exc)
            )
            import contextlib as _cl
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import (
                    tts_errors_total,
                    tts_fallback_total,
                )
                tts_errors_total.labels(provider=provider).inc()
                if label == "primary":
                    tts_fallback_total.inc()
            return False

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_messages(
        *,
        conversation_history: list[dict[str, str]],
        user_transcript: str,
        rag_context: str,
    ) -> list[dict[str, str]]:
        """Assemble the OpenAI-compatible message list for the LLM."""
        system_prompt = (
            "You are a helpful AI call center agent. "
            "Respond concisely and professionally."
        )
        if rag_context:
            system_prompt += (
                f"\n\nRelevant context from the knowledge base:\n{rag_context}"
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": user_transcript},
        ]
        return messages


def _utc_now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()
