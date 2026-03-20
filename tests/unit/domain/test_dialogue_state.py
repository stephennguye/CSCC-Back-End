"""Tests for src.domain.entities.dialogue_state."""

from __future__ import annotations

from src.domain.entities.dialogue_state import (
    BOOKING_SLOTS,
    REQUIRED_SLOTS,
    DialogueState,
    NLUResult,
    PolicyAction,
    PolicyDecision,
    SlotValue,
)


class TestDialogueStateCreation:
    def test_create_initialises_all_slots_to_none(self) -> None:
        state = DialogueState.create("s1")
        assert state.session_id == "s1"
        assert state.intent is None
        assert state.intent_confidence == 0.0
        assert state.confirmed is False
        assert state.turn_count == 0
        assert state.history == []
        for slot in BOOKING_SLOTS:
            assert state.slots[slot] is None

    def test_create_independent_instances(self) -> None:
        s1 = DialogueState.create("a")
        s2 = DialogueState.create("b")
        s1.slots["fromloc.city_name"] = "Hà Nội"
        assert s2.slots["fromloc.city_name"] is None


class TestFilledSlots:
    def test_empty_state_returns_empty_dict(self, fresh_state: DialogueState) -> None:
        assert fresh_state.filled_slots() == {}

    def test_returns_only_non_none_slots(self, fresh_state: DialogueState) -> None:
        fresh_state.slots["fromloc.city_name"] = "Hà Nội"
        fresh_state.slots["toloc.city_name"] = "Đà Nẵng"
        filled = fresh_state.filled_slots()
        assert filled == {
            "fromloc.city_name": "Hà Nội",
            "toloc.city_name": "Đà Nẵng",
        }


class TestMissingRequired:
    def test_all_missing_initially(self, fresh_state: DialogueState) -> None:
        assert fresh_state.missing_required() == REQUIRED_SLOTS

    def test_partially_filled(self, fresh_state: DialogueState) -> None:
        fresh_state.slots["fromloc.city_name"] = "Hà Nội"
        assert fresh_state.missing_required() == [
            "toloc.city_name", "depart_date", "depart_time.time",
            "airline_name", "class_type", "round_trip",
        ]

    def test_cities_and_date_still_needs_rest(self, fresh_state: DialogueState) -> None:
        fresh_state.slots["fromloc.city_name"] = "Hà Nội"
        fresh_state.slots["toloc.city_name"] = "Đà Nẵng"
        fresh_state.slots["depart_date.day_name"] = "thứ sáu"
        assert fresh_state.missing_required() == [
            "depart_time.time", "airline_name", "class_type", "round_trip",
        ]

    def test_all_filled_one_way(self, fresh_state: DialogueState) -> None:
        fresh_state.slots["fromloc.city_name"] = "Hà Nội"
        fresh_state.slots["toloc.city_name"] = "Đà Nẵng"
        fresh_state.slots["depart_date.day_name"] = "thứ sáu"
        fresh_state.slots["depart_time.time"] = "10 giờ sáng"
        fresh_state.slots["airline_name"] = "Vietnam Airlines"
        fresh_state.slots["class_type"] = "phổ thông"
        fresh_state.slots["round_trip"] = "một chiều"
        assert fresh_state.missing_required() == []

    def test_round_trip_requires_return_date(self, fresh_state: DialogueState) -> None:
        fresh_state.slots["fromloc.city_name"] = "Hà Nội"
        fresh_state.slots["toloc.city_name"] = "Đà Nẵng"
        fresh_state.slots["depart_date.day_name"] = "thứ sáu"
        fresh_state.slots["depart_time.time"] = "10 giờ sáng"
        fresh_state.slots["airline_name"] = "Vietnam Airlines"
        fresh_state.slots["class_type"] = "phổ thông"
        fresh_state.slots["round_trip"] = "khứ hồi"
        assert fresh_state.missing_required() == ["return_date"]

    def test_round_trip_all_filled(self, fresh_state: DialogueState) -> None:
        fresh_state.slots["fromloc.city_name"] = "Hà Nội"
        fresh_state.slots["toloc.city_name"] = "Đà Nẵng"
        fresh_state.slots["depart_date.day_name"] = "thứ sáu"
        fresh_state.slots["depart_time.time"] = "10 giờ sáng"
        fresh_state.slots["airline_name"] = "Vietnam Airlines"
        fresh_state.slots["class_type"] = "phổ thông"
        fresh_state.slots["round_trip"] = "khứ hồi"
        fresh_state.slots["return_date.day_name"] = "chủ nhật"
        assert fresh_state.missing_required() == []


class TestHasDateInfo:
    def test_no_date_info(self, fresh_state: DialogueState) -> None:
        assert fresh_state.has_date_info() is False

    def test_with_day_name(self, fresh_state: DialogueState) -> None:
        fresh_state.slots["depart_date.day_name"] = "thứ sáu"
        assert fresh_state.has_date_info() is True

    def test_with_month_name(self, fresh_state: DialogueState) -> None:
        fresh_state.slots["depart_date.month_name"] = "tháng 3"
        assert fresh_state.has_date_info() is True


class TestToDict:
    def test_serialization_round_trip(self, fresh_state: DialogueState) -> None:
        fresh_state.intent = "atis_flight"
        fresh_state.intent_confidence = 0.9
        fresh_state.turn_count = 2
        d = fresh_state.to_dict()
        assert d["session_id"] == "test-session"
        assert d["intent"] == "atis_flight"
        assert d["intent_confidence"] == 0.9
        assert d["turn_count"] == 2
        assert d["confirmed"] is False
        assert isinstance(d["slots"], dict)
        assert isinstance(d["history"], list)


class TestPolicyAction:
    def test_all_actions_have_string_values(self) -> None:
        for action in PolicyAction:
            assert isinstance(action.value, str)

    def test_expected_actions_exist(self) -> None:
        names = {a.name for a in PolicyAction}
        assert names == {
            "GREET", "CLARIFY", "REQUEST_SLOT", "CONFIRM",
            "EXECUTE", "PROVIDE_INFO", "ESCALATE", "FAREWELL",
        }


class TestSlotValue:
    def test_default_confidence(self) -> None:
        sv = SlotValue(name="x", value="y")
        assert sv.confidence == 1.0


class TestPolicyDecision:
    def test_defaults(self) -> None:
        pd = PolicyDecision(action=PolicyAction.GREET)
        assert pd.target_slot is None
        assert pd.params == {}
