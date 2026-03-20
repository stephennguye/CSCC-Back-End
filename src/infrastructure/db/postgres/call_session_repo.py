"""PostgreSQL implementation of CallSessionRepository.

Persists :class:`~src.domain.entities.call_session.CallSession` aggregates
and their associated :class:`~src.domain.entities.message.Message` records
using SQLAlchemy async ORM.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select

from src.domain.entities.call_session import CallSession
from src.domain.entities.message import Message
from src.domain.errors import PersistenceError, SessionNotFoundError
from src.domain.value_objects.session_state import SessionState
from src.domain.value_objects.speaker_role import SpeakerRole
from src.infrastructure.db.postgres.models import CallSessionModel, MessageModel

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


def _model_to_session(row: CallSessionModel) -> CallSession:
    return CallSession(
        id=row.id,
        state=SessionState(row.state),
        created_at=row.created_at,
        ended_at=row.ended_at,
        metadata=row.metadata_,
    )


def _model_to_message(row: MessageModel) -> Message:
    from src.domain.value_objects.confidence_score import ConfidenceScore

    confidence = (
        ConfidenceScore(value=row.confidence_score)
        if row.confidence_score is not None
        else None
    )
    return Message(
        id=row.id,
        session_id=row.session_id,
        role=SpeakerRole(row.role),
        content=row.content,
        timestamp=row.timestamp,
        sequence_number=row.sequence_number,
        confidence_score=confidence,
    )


class PostgresCallSessionRepository:
    """Async PostgreSQL-backed :class:`CallSessionRepository` implementation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # CallSession CRUD                                                     #
    # ------------------------------------------------------------------ #

    async def create(self, session: CallSession) -> CallSession:
        """Persist a newly created *CallSession*."""
        try:
            row = CallSessionModel(
                id=session.id,
                state=str(session.state),
                created_at=session.created_at,
                ended_at=session.ended_at,
                metadata_=session.metadata,
            )
            self._session.add(row)
            await self._session.flush()
            logger.debug("session_created", session_id=str(session.id))
            return session
        except Exception as exc:
            raise PersistenceError(f"Failed to create CallSession: {exc}") from exc

    async def get_by_id(self, session_id: uuid.UUID) -> CallSession | None:
        """Return the *CallSession* for *session_id*, or *None* if not found."""
        try:
            result = await self._session.execute(
                select(CallSessionModel).where(CallSessionModel.id == session_id)
            )
            row = result.scalar_one_or_none()
            return _model_to_session(row) if row is not None else None
        except Exception as exc:
            raise PersistenceError(
                f"Failed to fetch CallSession {session_id}: {exc}"
            ) from exc

    async def update_state(
        self,
        session_id: uuid.UUID,
        new_state: SessionState,
    ) -> CallSession:
        """Transition the session to *new_state* and persist ``ended_at`` when required.

        Raises:
            SessionNotFoundError: if *session_id* does not exist.
        """
        try:
            result = await self._session.execute(
                select(CallSessionModel).where(CallSessionModel.id == session_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise SessionNotFoundError(
                    f"Session {session_id} not found"
                )
            row.state = str(new_state)
            if new_state in (SessionState.ended, SessionState.error):
                row.ended_at = row.ended_at or datetime.now(UTC)
            await self._session.flush()
            return _model_to_session(row)
        except SessionNotFoundError:
            raise
        except Exception as exc:
            raise PersistenceError(
                f"Failed to update state for session {session_id}: {exc}"
            ) from exc

    async def update_metadata(
        self,
        session_id: uuid.UUID,
        metadata: dict,
    ) -> None:
        """Update the session's metadata JSONB column (merge with existing)."""
        try:
            result = await self._session.execute(
                select(CallSessionModel).where(CallSessionModel.id == session_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise SessionNotFoundError(f"Session {session_id} not found")
            existing = row.metadata_ or {}
            existing.update(metadata)
            row.metadata_ = existing
            await self._session.flush()
        except SessionNotFoundError:
            raise
        except Exception as exc:
            raise PersistenceError(
                f"Failed to update metadata for session {session_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Message operations                                                   #
    # ------------------------------------------------------------------ #

    async def append_message(self, message: Message) -> Message:
        """Insert a new *Message* record with auto-incremented sequence_number."""
        try:
            # Compute next sequence_number for this session
            result = await self._session.execute(
                select(func.coalesce(func.max(MessageModel.sequence_number), 0)).where(
                    MessageModel.session_id == message.session_id
                )
            )
            max_seq: int = result.scalar_one()  # type: ignore[assignment]
            next_seq = max_seq + 1

            row = MessageModel(
                id=message.id,
                session_id=message.session_id,
                role=str(message.role),
                content=message.content,
                confidence_score=(
                    message.confidence_score.value
                    if message.confidence_score is not None
                    else None
                ),
                timestamp=message.timestamp,
                sequence_number=next_seq,
            )
            self._session.add(row)
            await self._session.flush()
            # Return a new Message with the assigned sequence_number
            return Message(
                id=message.id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
                sequence_number=next_seq,
                confidence_score=message.confidence_score,
            )
        except Exception as exc:
            raise PersistenceError(f"Failed to append Message: {exc}") from exc

    async def bulk_append_messages(self, messages: list[Message]) -> list[Message]:
        """Insert multiple *Message* records with pre-computed sequence numbers.

        Gets the current max sequence_number once, then inserts all messages
        in a single flush — far more efficient than per-message SELECT MAX + INSERT.
        """
        if not messages:
            return []
        try:
            session_id = messages[0].session_id
            result = await self._session.execute(
                select(func.coalesce(func.max(MessageModel.sequence_number), 0)).where(
                    MessageModel.session_id == session_id
                )
            )
            max_seq: int = result.scalar_one()  # type: ignore[assignment]

            persisted: list[Message] = []
            for i, message in enumerate(messages, start=1):
                next_seq = max_seq + i
                row = MessageModel(
                    id=message.id,
                    session_id=message.session_id,
                    role=str(message.role),
                    content=message.content,
                    confidence_score=(
                        message.confidence_score.value
                        if message.confidence_score is not None
                        else None
                    ),
                    timestamp=message.timestamp,
                    sequence_number=next_seq,
                )
                self._session.add(row)
                persisted.append(
                    Message(
                        id=message.id,
                        session_id=message.session_id,
                        role=message.role,
                        content=message.content,
                        timestamp=message.timestamp,
                        sequence_number=next_seq,
                        confidence_score=message.confidence_score,
                    )
                )
            await self._session.flush()
            return persisted
        except Exception as exc:
            raise PersistenceError(f"Failed to bulk append Messages: {exc}") from exc

    async def list_messages_by_session(
        self,
        session_id: uuid.UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Message]:
        """Return ordered *Message* records for *session_id*."""
        try:
            result = await self._session.execute(
                select(MessageModel)
                .where(MessageModel.session_id == session_id)
                .order_by(MessageModel.sequence_number)
                .limit(limit)
                .offset(offset)
            )
            return [_model_to_message(row) for row in result.scalars().all()]
        except Exception as exc:
            raise PersistenceError(
                f"Failed to list messages for session {session_id}: {exc}"
            ) from exc

    async def count_messages_by_session(self, session_id: uuid.UUID) -> int:
        """Return total number of messages for *session_id*."""
        try:
            result = await self._session.execute(
                select(func.count(MessageModel.id)).where(
                    MessageModel.session_id == session_id
                )
            )
            return result.scalar_one() or 0
        except Exception as exc:
            raise PersistenceError(
                f"Failed to count messages for session {session_id}: {exc}"
            ) from exc
