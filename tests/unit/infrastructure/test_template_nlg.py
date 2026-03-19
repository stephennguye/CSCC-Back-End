"""Tests for src.infrastructure.nlg.template_nlg_adapter."""

from __future__ import annotations

from src.domain.entities.dialogue_state import (
    DialogueState,
    PolicyAction,
    PolicyDecision,
)
from src.infrastructure.nlg.template_nlg_adapter import TemplateNLGAdapter


class TestTemplateNLG:
    def setup_method(self) -> None:
        self.nlg = TemplateNLGAdapter()
        self.state = DialogueState.create("s1")

    def test_greet(self) -> None:
        d = PolicyDecision(action=PolicyAction.GREET)
        text = self.nlg.generate(d, self.state)
        assert "Xin chào" in text
        assert "trợ lý" in text

    def test_farewell(self) -> None:
        d = PolicyDecision(action=PolicyAction.FAREWELL)
        text = self.nlg.generate(d, self.state)
        assert "Cảm ơn" in text

    def test_request_slot_known(self) -> None:
        d = PolicyDecision(action=PolicyAction.REQUEST_SLOT, target_slot="fromloc.city_name")
        text = self.nlg.generate(d, self.state)
        assert "bay từ" in text or "thành phố" in text

    def test_request_slot_unknown(self) -> None:
        d = PolicyDecision(action=PolicyAction.REQUEST_SLOT, target_slot="some_new_slot")
        text = self.nlg.generate(d, self.state)
        assert "some_new_slot" in text

    def test_confirm_with_slots(self) -> None:
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        self.state.slots["toloc.city_name"] = "Đà Nẵng"
        self.state.slots["depart_date.day_name"] = "thứ sáu"
        d = PolicyDecision(action=PolicyAction.CONFIRM)
        text = self.nlg.generate(d, self.state)
        assert "Hà Nội" in text
        assert "Đà Nẵng" in text
        assert "thứ sáu" in text
        assert "xác nhận" in text

    def test_confirm_empty_slots(self) -> None:
        d = PolicyDecision(action=PolicyAction.CONFIRM)
        text = self.nlg.generate(d, self.state)
        assert "xác nhận" in text

    def test_execute(self) -> None:
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        self.state.slots["toloc.city_name"] = "Đà Nẵng"
        d = PolicyDecision(action=PolicyAction.EXECUTE)
        text = self.nlg.generate(d, self.state)
        assert "Hà Nội" in text
        assert "Đà Nẵng" in text

    def test_provide_info(self) -> None:
        d = PolicyDecision(action=PolicyAction.PROVIDE_INFO)
        text = self.nlg.generate(d, self.state)
        assert "thông tin" in text

    def test_escalate(self) -> None:
        d = PolicyDecision(action=PolicyAction.ESCALATE)
        text = self.nlg.generate(d, self.state)
        assert "nhân viên" in text

    def test_clarify(self) -> None:
        d = PolicyDecision(action=PolicyAction.CLARIFY)
        text = self.nlg.generate(d, self.state)
        assert "nói lại" in text

    def test_all_actions_produce_non_empty_response(self) -> None:
        for action in PolicyAction:
            d = PolicyDecision(action=action)
            text = self.nlg.generate(d, self.state)
            assert text, f"Empty response for {action}"
