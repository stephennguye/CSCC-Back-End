"""ReminderRepository — domain interface (Protocol).

Zero framework imports — pure Python typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid

    from src.domain.entities.reminder import Reminder


@runtime_checkable
class ReminderRepository(Protocol):
    """Persistence interface for *Reminder* records."""

    async def create(self, reminder: Reminder) -> Reminder:
        """Persist a new *Reminder* and return it."""
        ...  # pragma: no cover

    async def get_all_by_session_id(self, session_id: uuid.UUID) -> list[Reminder]:
        """Return all *Reminder* records for *session_id*.

        Returns an empty list (not an error) when none exist.
        """
        ...  # pragma: no cover
