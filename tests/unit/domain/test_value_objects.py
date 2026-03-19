"""Tests for domain value objects."""

from __future__ import annotations

import pytest

from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.session_state import SessionState, UrgencyLevel
from src.domain.value_objects.speaker_role import SpeakerRole


class TestConfidenceScore:
    def test_valid_score(self) -> None:
        cs = ConfidenceScore(value=0.85)
        assert cs.value == 0.85

    def test_zero_is_valid(self) -> None:
        cs = ConfidenceScore(value=0.0)
        assert cs.value == 0.0

    def test_one_is_valid(self) -> None:
        cs = ConfidenceScore(value=1.0)
        assert cs.value == 1.0

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ConfidenceScore(value=1.01)

    def test_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ConfidenceScore(value=-0.01)

    def test_is_below_threshold_true(self) -> None:
        cs = ConfidenceScore(value=0.3)
        assert cs.is_below_threshold(0.5) is True

    def test_is_below_threshold_false(self) -> None:
        cs = ConfidenceScore(value=0.7)
        assert cs.is_below_threshold(0.5) is False

    def test_is_below_threshold_at_boundary(self) -> None:
        cs = ConfidenceScore(value=0.5)
        assert cs.is_below_threshold(0.5) is False

    def test_frozen(self) -> None:
        cs = ConfidenceScore(value=0.5)
        with pytest.raises(AttributeError):
            cs.value = 0.9  # type: ignore[misc]


class TestSessionState:
    def test_all_states(self) -> None:
        assert SessionState.active.value == "active"
        assert SessionState.ended.value == "ended"
        assert SessionState.error.value == "error"

    def test_is_str_enum(self) -> None:
        assert isinstance(SessionState.active, str)
        assert SessionState.active == "active"


class TestUrgencyLevel:
    def test_all_levels(self) -> None:
        assert UrgencyLevel.low.value == "low"
        assert UrgencyLevel.medium.value == "medium"
        assert UrgencyLevel.high.value == "high"
        assert UrgencyLevel.critical.value == "critical"

    def test_ordering_by_name(self) -> None:
        levels = [UrgencyLevel.critical, UrgencyLevel.low, UrgencyLevel.high, UrgencyLevel.medium]
        names = {level.name for level in levels}
        assert names == {"low", "medium", "high", "critical"}


class TestSpeakerRole:
    def test_all_roles(self) -> None:
        assert SpeakerRole.user.value == "user"
        assert SpeakerRole.ai.value == "ai"

    def test_is_str_enum(self) -> None:
        assert isinstance(SpeakerRole.user, str)
        assert SpeakerRole.user == "user"
