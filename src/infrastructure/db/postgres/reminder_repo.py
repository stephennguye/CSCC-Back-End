"""PostgreSQL implementation of ReminderRepository.

Persists :class:`~src.domain.entities.reminder.Reminder` records using
SQLAlchemy async ORM.

Multiple reminders per session are permitted (no unique constraint on
session_id).  ``get_all_by_session_id`` returns an empty list — not an error —
when no reminders exist for the given session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select

from src.domain.entities.reminder import Reminder
from src.domain.errors import PersistenceError
from src.infrastructure.db.postgres.models import ReminderModel

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# ORM ↔ domain mapping helpers
# ──────────────────────────────────────────────────────────────────────────────


def _model_to_reminder(row: ReminderModel) -> Reminder:
    return Reminder(
        id=row.id,
        session_id=row.session_id,
        description=row.description,
        target_due_at=row.target_due_at,
        created_at=row.created_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Repository
# ──────────────────────────────────────────────────────────────────────────────


class PostgresReminderRepository:
    """Async PostgreSQL-backed :class:`ReminderRepository` implementation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # CRUD                                                                 #
    # ------------------------------------------------------------------ #

    async def create(self, reminder: Reminder) -> Reminder:
        """Persist a new *Reminder* and return it.

        Raises:
            PersistenceError: on database error.
        """
        try:
            row = ReminderModel(
                id=reminder.id,
                session_id=reminder.session_id,
                description=reminder.description,
                target_due_at=reminder.target_due_at,
                created_at=reminder.created_at,
            )
            self._session.add(row)
            await self._session.flush()
            logger.debug(
                "reminder_created",
                reminder_id=str(reminder.id),
                session_id=str(reminder.session_id),
            )
            return reminder
        except Exception as exc:
            logger.error(
                "reminder_create_failed",
                session_id=str(reminder.session_id),
                error=str(exc),
            )
            raise PersistenceError(f"Failed to create Reminder: {exc}") from exc

    async def get_all_by_session_id(
        self, session_id: uuid.UUID
    ) -> list[Reminder]:
        """Return all *Reminder* records for *session_id*.

        Returns an empty list (not an error) when none exist.
        """
        try:
            stmt = select(ReminderModel).where(
                ReminderModel.session_id == session_id
            )
            result = await self._session.execute(stmt)
            rows = result.scalars().all()
            reminders = [_model_to_reminder(row) for row in rows]
            logger.debug(
                "reminders_fetched",
                session_id=str(session_id),
                count=len(reminders),
            )
            return reminders
        except Exception as exc:
            logger.error(
                "reminder_fetch_failed",
                session_id=str(session_id),
                error=str(exc),
            )
            raise PersistenceError(
                f"Failed to fetch Reminders for session {session_id}: {exc}"
            ) from exc
