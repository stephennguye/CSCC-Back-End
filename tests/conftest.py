"""Shared test fixtures."""

from __future__ import annotations

import pytest

from src.domain.entities.dialogue_state import (
    BOOKING_SLOTS,
    DialogueState,
    NLUResult,
    PolicyAction,
    PolicyDecision,
    SlotValue,
)


@pytest.fixture()
def fresh_state() -> DialogueState:
    """Return a blank dialogue state for session 'test-session'."""
    return DialogueState.create("test-session")


@pytest.fixture()
def flight_nlu_result() -> NLUResult:
    """Return an NLU result for a typical flight booking utterance."""
    return NLUResult(
        intent="atis_flight",
        intent_confidence=0.92,
        slots=[
            SlotValue(name="fromloc.city_name", value="Hà Nội", confidence=0.85),
            SlotValue(name="toloc.city_name", value="Đà Nẵng", confidence=0.85),
            SlotValue(name="depart_date.day_name", value="thứ sáu", confidence=0.85),
        ],
        raw_text="Tôi muốn bay từ Hà Nội đến Đà Nẵng ngày thứ sáu",
    )
