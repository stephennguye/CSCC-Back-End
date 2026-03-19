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

    def test_all_required_filled_confirms(self) -> None:
        self.state.intent = "atis_flight"
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        self.state.slots["toloc.city_name"] = "Đà Nẵng"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.CONFIRM

    def test_confirmed_executes(self) -> None:
        self.state.intent = "atis_flight"
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        self.state.slots["toloc.city_name"] = "Đà Nẵng"
        self.state.confirmed = True
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.EXECUTE

    def test_full_booking_flow(self) -> None:
        """Simulate a complete booking: request → request → confirm → execute."""
        # Turn 1: no slots
        self.state.intent = "atis_flight"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.REQUEST_SLOT

        # Turn 2: fromloc filled
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.REQUEST_SLOT

        # Turn 3: toloc filled
        self.state.slots["toloc.city_name"] = "Đà Nẵng"
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.CONFIRM

        # Turn 4: user confirms
        self.state.confirmed = True
        d = self.policy.decide(self.state)
        assert d.action == PolicyAction.EXECUTE
