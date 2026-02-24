"""CallSessionRepository — domain interface (Protocol).

Zero framework imports — pure Python typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid

    from src.domain.entities.call_session import CallSession
    from src.domain.entities.message import Message
    from src.domain.value_objects.session_state import SessionState


@runtime_checkable
class CallSessionRepository(Protocol):
    """Persistence interface for *CallSession* and *Message* aggregates."""

    # ------------------------------------------------------------------ #
    # CallSession CRUD                                                     #
    # ------------------------------------------------------------------ #

    async def create(self, session: CallSession) -> CallSession:
        """Persist a newly created *CallSession* and return it."""
        ...  # pragma: no cover

    async def get_by_id(self, session_id: uuid.UUID) -> CallSession | None:
        """Return the *CallSession* for *session_id*, or *None* if not found."""
        ...  # pragma: no cover

    async def update_state(
        self,
        session_id: uuid.UUID,
        new_state: SessionState,
    ) -> CallSession:
        """Transition *session_id* to *new_state* and set ``ended_at`` when required.

        Raises:
            SessionNotFoundError: if *session_id* does not exist.
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------ #
    # Message operations                                                  #
    # ------------------------------------------------------------------ #

    async def append_message(self, message: Message) -> Message:
        """Insert a new *Message* record.

        ``message.sequence_number`` is auto-assigned by the repository
        implementation to ensure monotonic ordering within the session.
        """
        ...  # pragma: no cover

    async def list_messages_by_session(
        self,
        session_id: uuid.UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Message]:
        """Return an ordered list of *Message* records for *session_id*."""
        ...  # pragma: no cover

    async def count_messages_by_session(self, session_id: uuid.UUID) -> int:
        """Return the total number of messages for *session_id*."""
        ...  # pragma: no cover
