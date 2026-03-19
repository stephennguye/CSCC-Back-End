"""Tests for src.infrastructure.dst.hybrid_dst_adapter."""

from __future__ import annotations

from src.domain.entities.dialogue_state import DialogueState, NLUResult, SlotValue
from src.infrastructure.dst.hybrid_dst_adapter import HybridDSTAdapter


class TestHybridDST:
    def setup_method(self) -> None:
        self.dst = HybridDSTAdapter()
        self.state = DialogueState.create("s1")

    def test_updates_intent(self) -> None:
        nlu = NLUResult(intent="atis_flight", intent_confidence=0.9)
        updated = self.dst.update(self.state, nlu)
        assert updated.intent == "atis_flight"
        assert updated.intent_confidence == 0.9

    def test_increments_turn_count(self) -> None:
        nlu = NLUResult(intent="atis_flight", intent_confidence=0.9)
        updated = self.dst.update(self.state, nlu)
        assert updated.turn_count == 1
        updated = self.dst.update(updated, nlu)
        assert updated.turn_count == 2

    def test_merges_slots_above_threshold(self) -> None:
        nlu = NLUResult(
            intent="atis_flight",
            intent_confidence=0.9,
            slots=[
                SlotValue(name="fromloc.city_name", value="Hà Nội", confidence=0.85),
                SlotValue(name="toloc.city_name", value="Đà Nẵng", confidence=0.85),
            ],
        )
        updated = self.dst.update(self.state, nlu)
        assert updated.slots["fromloc.city_name"] == "Hà Nội"
        assert updated.slots["toloc.city_name"] == "Đà Nẵng"

    def test_rejects_slots_below_threshold(self) -> None:
        nlu = NLUResult(
            intent="atis_flight",
            intent_confidence=0.9,
            slots=[
                SlotValue(name="fromloc.city_name", value="Hà Nội", confidence=0.3),
            ],
        )
        updated = self.dst.update(self.state, nlu)
        assert updated.slots["fromloc.city_name"] is None

    def test_overwrites_slot_with_higher_confidence(self) -> None:
        self.state.slots["fromloc.city_name"] = "Hà Nội"
        nlu = NLUResult(
            intent="atis_flight",
            intent_confidence=0.9,
            slots=[
                SlotValue(name="fromloc.city_name", value="Hồ Chí Minh", confidence=0.9),
            ],
        )
        updated = self.dst.update(self.state, nlu)
        assert updated.slots["fromloc.city_name"] == "Hồ Chí Minh"

    def test_records_history(self) -> None:
        nlu = NLUResult(
            intent="atis_flight",
            intent_confidence=0.9,
            slots=[SlotValue(name="fromloc.city_name", value="Hà Nội", confidence=0.85)],
        )
        updated = self.dst.update(self.state, nlu)
        assert len(updated.history) == 1
        assert updated.history[0]["turn"] == 1
        assert updated.history[0]["intent"] == "atis_flight"
        assert updated.history[0]["slots"] == {"fromloc.city_name": "Hà Nội"}

    def test_preserves_existing_slots_across_turns(self) -> None:
        nlu1 = NLUResult(
            intent="atis_flight",
            intent_confidence=0.9,
            slots=[SlotValue(name="fromloc.city_name", value="Hà Nội", confidence=0.85)],
        )
        nlu2 = NLUResult(
            intent="atis_flight",
            intent_confidence=0.9,
            slots=[SlotValue(name="toloc.city_name", value="Đà Nẵng", confidence=0.85)],
        )
        state = self.dst.update(self.state, nlu1)
        state = self.dst.update(state, nlu2)
        assert state.slots["fromloc.city_name"] == "Hà Nội"
        assert state.slots["toloc.city_name"] == "Đà Nẵng"

    def test_exact_threshold_accepted(self) -> None:
        nlu = NLUResult(
            intent="atis_flight",
            intent_confidence=0.9,
            slots=[SlotValue(name="fromloc.city_name", value="Hà Nội", confidence=0.5)],
        )
        updated = self.dst.update(self.state, nlu)
        assert updated.slots["fromloc.city_name"] == "Hà Nội"
