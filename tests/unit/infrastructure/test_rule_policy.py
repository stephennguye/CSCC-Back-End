"""Tests for src.infrastructure.policy.rule_policy_adapter."""

from __future__ import annotations

from src.domain.entities.dialogue_state import DialogueState, PolicyAction
from src.infrastructure.policy.rule_policy_adapter import RulePolicyAdapter


class TestRulePolicy:
    def setup_method(self) -> None:
        self.policy = RulePolicyAdapter()
        self.state = DialogueState.create("s1")

    def test_greet_intent(self) -> None:
        self.state.intent = "greet"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.GREET

    def test_farewell_intent(self) -> None:
        self.state.intent = "farewell"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.FAREWELL

    def test_abbreviation_intent_clarifies(self) -> None:
        self.state.intent = "atis_abbreviation"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.CLARIFY

    def test_ground_service_intent_clarifies(self) -> None:
        self.state.intent = "atis_ground_service"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.CLARIFY

    def test_missing_required_slots_requests_first(self) -> None:
        self.state.intent = "atis_flight"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.REQUEST_SLOT
        assert d.target_slot == "fromloc.city_name"

    def test_one_required_filled_requests_second(self) -> None:
        self.state.intent = "atis_flight"
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.REQUEST_SLOT
        assert d.target_slot == "toloc.city_name"

    def test_cities_filled_but_no_date_requests_date(self) -> None:
        self.state.intent = "atis_flight"
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        self.state.slots["toloc.city_name"] = "Đà Nẵng"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.REQUEST_SLOT
        assert d.target_slot == "depart_date"

    def _fill_all_required(self) -> None:
        """Helper to fill all required slots (one-way booking)."""
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        self.state.slots["toloc.city_name"] = "Đà Nẵng"
        self.state.slots["depart_date.day_name"] = "thứ sáu"
        self.state.slots["depart_time.time"] = "10 giờ sáng"
        self.state.slots["airline_name"] = "Vietnam Airlines"
        self.state.slots["class_type"] = "phổ thông"
        self.state.slots["round_trip"] = "một chiều"

    def test_all_required_filled_confirms(self) -> None:
        self.state.intent = "atis_flight"
        self._fill_all_required()
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.CONFIRM

    def test_confirmed_executes(self) -> None:
        self.state.intent = "atis_flight"
        self._fill_all_required()
        self.state.confirmed = True
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.EXECUTE

    def test_full_booking_flow_one_way(self) -> None:
        """Simulate: gather every slot (one-way) → confirm → execute."""
        self.state.intent = "atis_flight"

        steps = [
            ("fromloc.city_name", "Hà Nội", "toloc.city_name"),
            ("toloc.city_name", "Đà Nẵng", "depart_date"),
            ("depart_date.day_name", "thứ sáu", "depart_time.time"),
            ("depart_time.time", "10 giờ sáng", "airline_name"),
            ("airline_name", "Vietnam Airlines", "class_type"),
            ("class_type", "phổ thông", "round_trip"),
        ]

        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.REQUEST_SLOT
        assert d.target_slot == "fromloc.city_name"

        for slot_key, slot_val, next_slot in steps:
            self.state.slots[slot_key] = slot_val
            d = self.policy.decide(self.state)
            assert d.action == PolicyAction.REQUEST_SLOT
            assert d.target_slot == next_slot

        # One-way → all filled → confirm
        self.state.slots["round_trip"] = "một chiều"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.CONFIRM

        self.state.confirmed = True
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.EXECUTE

    def test_full_booking_flow_round_trip(self) -> None:
        """Simulate: round-trip requires return_date before confirm."""
        self.state.intent = "atis_flight"
        self._fill_all_required()
        # Change to round-trip → needs return_date
        self.state.slots["round_trip"] = "khứ hồi"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.REQUEST_SLOT
        assert d.target_slot == "return_date"

        # Fill return date → confirm
        self.state.slots["return_date.day_name"] = "chủ nhật"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.CONFIRM
