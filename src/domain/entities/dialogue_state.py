"""Dialogue state entity for task-oriented dialogue."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class PolicyAction(Enum):
    """Actions the policy module can take."""

    GREET = "greet"
    CLARIFY = "clarify"
    REQUEST_SLOT = "request_slot"
    CONFIRM = "confirm"
    EXECUTE = "execute"
    PROVIDE_INFO = "provide_info"
    ESCALATE = "escalate"
    FAREWELL = "farewell"


@dataclass
class SlotValue:
    """A single extracted slot from NLU."""

    name: str
    value: str
    confidence: float = 1.0


@dataclass
class NLUResult:
    """Output from the NLU module."""

    intent: str
    intent_confidence: float
    slots: list[SlotValue] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class PolicyDecision:
    """Output from the policy module."""

    action: PolicyAction
    target_slot: str | None = None
    params: dict[str, str] = field(default_factory=dict)


BOOKING_SLOTS: list[str] = [
    "fromloc.city_name",
    "toloc.city_name",
    "depart_date.day_name",
    "depart_date.month_name",
    "depart_date.day_number",
    "depart_date.today_relative",
    "depart_time.time",
    "airline_name",
    "flight_number",
    "class_type",
    "round_trip",
]

REQUIRED_SLOTS: list[str] = [
    "fromloc.city_name",
    "toloc.city_name",
]


@dataclass
class DialogueState:
    """Belief state for airline booking dialogue.

    Tracks accumulated slot values across conversation turns.
    """

    session_id: str
    intent: str | None = None
    intent_confidence: float = 0.0
    slots: dict[str, str | None] = field(
        default_factory=lambda: {slot: None for slot in BOOKING_SLOTS}
    )
    confirmed: bool = False
    executed: bool = False
    turn_count: int = 0
    history: list[dict[str, object]] = field(default_factory=list)

    @staticmethod
    def create(session_id: str) -> DialogueState:
        """Factory method to create a new dialogue state."""
        return DialogueState(session_id=session_id)

    def filled_slots(self) -> dict[str, str]:
        """Return only slots that have values."""
        return {k: v for k, v in self.slots.items() if v is not None}

    def missing_required(self) -> list[str]:
        """Return required slots that are not yet filled."""
        return [s for s in REQUIRED_SLOTS if not self.slots.get(s)]

    def has_date_info(self) -> bool:
        """Check if at least one date-related slot is filled."""
        date_slots = [
            "depart_date.day_name",
            "depart_date.month_name",
            "depart_date.day_number",
        ]
        return any(self.slots.get(s) for s in date_slots)

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict for API responses."""
        return {
            "session_id": self.session_id,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "slots": dict(self.slots),
            "confirmed": self.confirmed,
            "turn_count": self.turn_count,
            "history": list(self.history),
        }

    def reset_for_new_booking(self) -> None:
        """Reset state for a new booking while keeping session context."""
        self.intent = None
        self.intent_confidence = 0.0
        self.slots = {slot: None for slot in BOOKING_SLOTS}
        self.confirmed = False
        self.executed = False
