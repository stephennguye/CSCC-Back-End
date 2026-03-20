from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.domain.value_objects.session_state import SessionState


@dataclass
class CallSession:
    """Aggregate root for a single call session."""

    id: uuid.UUID
    state: SessionState
    created_at: datetime
    ended_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        *,
        session_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CallSession:
        """Create a new active *CallSession*."""
        return cls(
            id=session_id or uuid.uuid4(),
            state=SessionState.active,
            created_at=datetime.now(UTC),
            ended_at=None,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # State transitions                                                    #
    # ------------------------------------------------------------------ #

    def end(self, ended_at: datetime | None = None) -> None:
        """Transition the session to *ended* state."""
        self._validate_active("end")
        self.state = SessionState.ended
        self.ended_at = ended_at or datetime.now(UTC)

    def mark_error(self, ended_at: datetime | None = None) -> None:
        """Transition the session to *error* state."""
        self._validate_active("mark_error")
        self.state = SessionState.error
        self.ended_at = ended_at or datetime.now(UTC)

    # ------------------------------------------------------------------ #
    # Invariants                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate_ended_at()

    def _validate_ended_at(self) -> None:
        if self.state == SessionState.active and self.ended_at is not None:
            raise ValueError("ended_at must be null when state is active")
        if self.state in (SessionState.ended, SessionState.error) and self.ended_at is None:
            raise ValueError("ended_at must be set when state is ended or error")

    def _validate_active(self, operation: str) -> None:
        if self.state != SessionState.active:
            raise ValueError(
                f"Cannot perform '{operation}' on a session in state '{self.state.value}'"
            )
