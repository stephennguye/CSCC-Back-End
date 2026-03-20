from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.value_objects.session_state import UrgencyLevel


@dataclass
class Claim:
    """Structured post-call claim extracted from a completed *CallSession*.

    Unresolvable fields MUST remain *None* — never guessed (FR-013).
    The ``session_id`` is unique: at most one *Claim* per *CallSession*.
    """

    id: uuid.UUID
    session_id: uuid.UUID
    extracted_at: datetime
    schema_version: str = "v1"

    # Nullable — null when not determinable from transcript
    student_name: str | None = None
    issue_category: str | None = None
    urgency_level: UrgencyLevel | None = None
    confidence: float | None = None
    requested_action: str | None = None
    follow_up_date: date | None = None

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        *,
        session_id: uuid.UUID,
        student_name: str | None = None,
        issue_category: str | None = None,
        urgency_level: UrgencyLevel | None = None,
        confidence: float | None = None,
        requested_action: str | None = None,
        follow_up_date: date | None = None,
        schema_version: str = "v1",
        claim_id: uuid.UUID | None = None,
        extracted_at: datetime | None = None,
    ) -> Claim:
        return cls(
            id=claim_id or uuid.uuid4(),
            session_id=session_id,
            extracted_at=extracted_at or datetime.now(UTC),
            schema_version=schema_version,
            student_name=student_name,
            issue_category=issue_category,
            urgency_level=urgency_level,
            confidence=confidence,
            requested_action=requested_action,
            follow_up_date=follow_up_date,
        )

    # ------------------------------------------------------------------ #
    # Invariants                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Claim.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
