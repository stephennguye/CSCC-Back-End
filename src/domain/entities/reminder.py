from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class Reminder:
    """An actionable follow-up item derived from a completed *CallSession*.

    Multiple reminders per session are permitted.
    ``target_due_at`` is null when no parseable date/time was detected (FR-018).
    """

    id: uuid.UUID
    session_id: uuid.UUID
    description: str
    created_at: datetime
    target_due_at: datetime | None = None

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        *,
        session_id: uuid.UUID,
        description: str,
        target_due_at: datetime | None = None,
        reminder_id: uuid.UUID | None = None,
        created_at: datetime | None = None,
    ) -> Reminder:
        return cls(
            id=reminder_id or uuid.uuid4(),
            session_id=session_id,
            description=description,
            created_at=created_at or datetime.now(UTC),
            target_due_at=target_due_at,
        )

    # ------------------------------------------------------------------ #
    # Invariants                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError("Reminder.description must not be empty")
