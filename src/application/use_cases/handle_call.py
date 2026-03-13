"""HandleCallUseCase — top-level orchestrator for a real-time call session.

Responsibilities:
  1. Create a *CallSession* in PostgreSQL and mark the client as present in Redis.
  2. Accept streaming audio chunks into a per-session asyncio Queue.
  3. When ``audio.end`` is received, drain the Queue and pass the audio stream
     to :class:`~src.application.use_cases.stream_conversation.StreamConversationUseCase`.
  4. Detect barge-in: if the caller sends new audio while the AI is generating
     a response, publish a cancellation signal to the Redis Pub/Sub channel.
  5. On ``session.end`` or abrupt WebSocket disconnect, flush in-flight TTS,
     persist all Message records to PostgreSQL, transition the *CallSession*
     to *ended*, and enqueue background tasks (claim extraction, reminder
     generation — Phase 3 / Phase 5 hooks).
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.entities.call_session import CallSession
from src.domain.errors import SessionNotFoundError
from src.domain.value_objects.session_state import SessionState

if TYPE_CHECKING:

    from fastapi import BackgroundTasks

    from src.application.use_cases.extract_claims import ExtractClaimsUseCase
    from src.application.use_cases.generate_reminder import GenerateReminderUseCase
    from src.application.use_cases.stream_conversation import (
        SendBinaryFn,
        SendTextFn,
        StreamConversationUseCase,
    )
    from src.application.use_cases.tod_pipeline import TODPipelineUseCase
    from src.domain.entities.message import Message
    from src.infrastructure.cache.redis_client import RedisClient

logger = structlog.get_logger(__name__)

# ── OpenTelemetry helpers ────────────────────────────────────────────────────


def _get_tracer():  # type: ignore[return]  # noqa: ANN202
    """Return an OTel tracer, or a no-op context manager factory on ImportError."""
    try:
        from opentelemetry import trace  # type: ignore[import-untyped]

        return trace.get_tracer("cscc.handle_call")
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


# ── Sentinel to signal end-of-audio-stream ────────────────────────────────────
_AUDIO_END = b""

# Type alias for async session factory
SessionFactory = Callable[[], Any]  # returns async context manager yielding AsyncSession


class HandleCallUseCase:
    """Orchestrates the full lifecycle of one WebSocket call session.

    The ``session_factory`` is called for each database operation to get a
    fresh :class:`AsyncSession`, keeping the use case safe to use as a
    long-lived singleton across many concurrent WebSocket sessions.
    """

    def __init__(
        self,
        session_factory: Any,  # async_sessionmaker[AsyncSession]  # noqa: ANN401
        redis: RedisClient,
        stream_conversation: StreamConversationUseCase,
        extract_claims: ExtractClaimsUseCase | None = None,
        generate_reminders: GenerateReminderUseCase | None = None,
        tod_pipeline: TODPipelineUseCase | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._redis = redis
        self._stream = stream_conversation
        self._extract_claims = extract_claims
        self._generate_reminders = generate_reminders
        self._tod_pipeline = tod_pipeline

        # Per-session audio queues
        # session_id → asyncio.Queue[bytes | None]
        self._audio_queues: dict[str, asyncio.Queue[bytes]] = {}

        # Per-session collected messages (to be persisted at teardown)
        self._pending_messages: dict[str, list[Message]] = {}

        # Track whether AI is currently generating a response
        self._ai_responding: set[str] = set()

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    async def create_session(
        self,
        session_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CallSession:
        """Create a new *CallSession* in PostgreSQL and mark presence in Redis.

        Returns the persisted *CallSession*.
        """
        tracer = _get_tracer()
        with tracer.start_as_current_span("call.start") as span:
            session = CallSession.create(
                session_id=session_id,
                metadata=metadata,
            )
            async with self._session_factory() as db_session:
                from src.infrastructure.db.postgres.call_session_repo import (
                    PostgresCallSessionRepository,
                )
                repo = PostgresCallSessionRepository(db_session)
                session = await repo.create(session)
                await db_session.commit()

            session_str = str(session.id)
            span.set_attribute("session_id", session_str)

            await self._redis.mark_present(session_str)
            await self._redis.set_turn_state(session_str, "idle")

            # Initialise per-session state
            self._audio_queues[session_str] = asyncio.Queue()
            self._pending_messages[session_str] = []

            logger.info("session_created", session_id=session_str)
            return session

    async def get_or_create_token_session(
        self,
        session_id: uuid.UUID,
    ) -> CallSession:
        """Retrieve an existing active session by ID for token re-issuance.

        Raises :class:`~src.domain.errors.SessionNotFoundError` when the
        session does not exist or is not active.
        """
        async with self._session_factory() as db_session:
            from src.infrastructure.db.postgres.call_session_repo import (
                PostgresCallSessionRepository,
            )
            repo = PostgresCallSessionRepository(db_session)
            session = await repo.get_by_id(session_id)
        if session is None or session.state != SessionState.active:
            raise SessionNotFoundError(
                f"No active session found with id={session_id}"
            )
        return session

    # ------------------------------------------------------------------ #
    # Audio frame handling                                                 #
    # ------------------------------------------------------------------ #

    async def handle_audio_chunk(
        self,
        session_id: str,
        audio_bytes: bytes,
    ) -> None:
        """Enqueue raw audio bytes and detect barge-in.

        If the AI is currently generating a response, publish a barge-in
        cancellation signal to Redis.
        """
        if session_id in self._ai_responding:
            logger.debug("barge_in_triggered", session_id=session_id)
            import contextlib as _cl
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import ws_bargein_total
                ws_bargein_total.inc()
            await self._redis.publish_barge_in(session_id)

        queue = self._audio_queues.get(session_id)
        if queue is None:
            logger.warning("audio_chunk_no_queue", session_id=session_id)
            return
        await queue.put(audio_bytes)

    async def handle_audio_end(
        self,
        session_id: str,
        send_text: SendTextFn,
        send_binary: SendBinaryFn,
    ) -> None:
        """Signal end-of-audio and run the STT→LLM→TTS pipeline.

        Signals the audio queue with the sentinel value, then runs
        :meth:`StreamConversationUseCase.run` with the accumulated audio
        and conversations history from Redis.
        """
        queue = self._audio_queues.get(session_id)
        if queue is None:
            logger.warning("audio_end_no_queue", session_id=session_id)
            return

        # Signal end-of-stream
        await queue.put(_AUDIO_END)

        audio_gen = _drain_queue(queue)

        # Read conversation history from Redis buffer
        conversation_history = await self._build_conversation_history(session_id)

        self._ai_responding.add(session_id)
        try:
            from opentelemetry import baggage, context  # type: ignore[import-untyped]
            ctx = baggage.set_baggage("session_id", session_id)
            otel_token = context.attach(ctx)
        except ImportError:
            otel_token = None

        try:
            ai_text = await self._stream.run(
                session_id=uuid.UUID(session_id),
                audio_stream=audio_gen,
                conversation_history=conversation_history,
                send_text=send_text,
                send_binary=send_binary,
            )
        finally:
            self._ai_responding.discard(session_id)
            if otel_token is not None:
                with contextlib.suppress(Exception):
                    from opentelemetry import context as otel_ctx  # type: ignore[import-untyped]
                    otel_ctx.detach(otel_token)

        # Re-create a fresh queue for the next turn
        self._audio_queues[session_id] = asyncio.Queue()

        if ai_text:
            # Buffer the AI response for context in the next turn
            import json

            await self._redis.push_to_buffer(
                session_id,
                json.dumps({"role": "assistant", "content": ai_text}),
            )

    # ------------------------------------------------------------------ #
    # Session teardown                                                     #
    # ------------------------------------------------------------------ #

    async def teardown(
        self,
        session_id: str,
        background_tasks: BackgroundTasks | None = None,
        *,
        state: SessionState = SessionState.ended,
    ) -> None:
        """Flush, persist messages, transition session state.

        This method is safe to call multiple times; it is idempotent after
        first completion.
        """
        logger.info("session_teardown", session_id=session_id, new_state=state)

        tracer = _get_tracer()
        with tracer.start_as_current_span("call.teardown") as span:
            span.set_attribute("session_id", session_id)
            fin = state.value if hasattr(state, "value") else str(state)
            span.set_attribute("final_state", fin)

            # Cancel any in-progress AI response
            if session_id in self._ai_responding:
                await self._redis.publish_barge_in(session_id)
                self._ai_responding.discard(session_id)

            # Remove presence marker
            await self._redis.mark_absent(session_id)

            # Emit WS disconnection metric
            import contextlib as _cl
            with _cl.suppress(Exception):
                from src.infrastructure.observability.metrics import ws_disconnections_total
                ws_disconnections_total.labels(reason="clean").inc()

            # Persist all buffered messages and transition state in one session
            pending = self._pending_messages.pop(session_id, [])
            try:
                async with self._session_factory() as db_session:
                    from src.infrastructure.db.postgres.call_session_repo import (
                        PostgresCallSessionRepository,
                    )
                    repo = PostgresCallSessionRepository(db_session)
                    for message in pending:
                        try:
                            await repo.append_message(message)
                        except Exception:
                            logger.exception("message_persist_error", session_id=session_id)
                    try:
                        await repo.update_state(uuid.UUID(session_id), state)
                    except SessionNotFoundError:
                        logger.warning("teardown_session_not_found", session_id=session_id)
                    await db_session.commit()
            except Exception:
                logger.exception("teardown_db_error", session_id=session_id)

            # Enqueue background tasks (claim extraction, reminders — Phase 5/6)
            if background_tasks is not None and self._extract_claims is not None:
                # T047 — Claim extraction (non-blocking; MUST NOT await result)
                background_tasks.add_task(
                    self._extract_claims.execute,
                    uuid.UUID(session_id),
                )
                logger.debug("claim_extraction_enqueued", session_id=session_id)
                # T052 — Reminder generation (non-blocking; independent of claim extraction)
            if background_tasks is not None and self._generate_reminders is not None:
                background_tasks.add_task(
                    self._generate_reminders.execute,
                    uuid.UUID(session_id),
                )
                logger.debug("reminder_generation_enqueued", session_id=session_id)

            # Cleanup per-session state
            self._audio_queues.pop(session_id, None)

        logger.info("session_teardown_complete", session_id=session_id)

    # ------------------------------------------------------------------ #
    # Message buffering helpers                                            #
    # ------------------------------------------------------------------ #

    def buffer_message(self, session_id: str, message: Message) -> None:
        """Buffer a *Message* for batch persistence at teardown time."""
        buf = self._pending_messages.setdefault(session_id, [])
        buf.append(message)

    async def _build_conversation_history(
        self, session_id: str
    ) -> list[dict[str, str]]:
        """Read the Redis short-term buffer and deserialise into message dicts."""
        import json

        turns = await self._redis.get_buffer(session_id)
        history: list[dict[str, str]] = []
        for turn_json in turns:
            try:
                history.append(json.loads(turn_json))
            except Exception:
                logger.warning("buffer_parse_error", session_id=session_id)
        return history


# ── Audio queue async generator ───────────────────────────────────────────────


async def _drain_queue(queue: asyncio.Queue[bytes]) -> AsyncGenerator[bytes, None]:
    """Yield bytes from *queue* until the sentinel empty-bytes value is received."""
    while True:
        chunk = await queue.get()
        if chunk == _AUDIO_END:
            return
        yield chunk
