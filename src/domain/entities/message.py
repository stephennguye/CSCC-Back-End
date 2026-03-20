from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.value_objects.confidence_score import ConfidenceScore
    from src.domain.value_objects.speaker_role import SpeakerRole


@dataclass
class Message:
    """A single conversation turn within a *CallSession*."""

    id: uuid.UUID
    session_id: uuid.UUID
    role: SpeakerRole
    content: str
    timestamp: datetime
    sequence_number: int
    confidence_score: ConfidenceScore | None = None

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        *,
        session_id: uuid.UUID,
        role: SpeakerRole,
        content: str,
        sequence_number: int,
        confidence_score: ConfidenceScore | None = None,
        timestamp: datetime | None = None,
        message_id: uuid.UUID | None = None,
    ) -> Message:
        return cls(
            id=message_id or uuid.uuid4(),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=timestamp or datetime.now(UTC),
            sequence_number=sequence_number,
            confidence_score=confidence_score,
        )

    # ------------------------------------------------------------------ #
    # Invariants                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Message.content must not be empty")
        if self.sequence_number <= 0:
            raise ValueError("Message.sequence_number must be a positive integer")
