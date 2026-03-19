"""Tests for src.application.use_cases.tod_pipeline."""

from __future__ import annotations

import pytest

from src.domain.entities.dialogue_state import (
    DialogueState,
    NLUResult,
    PolicyAction,
    PolicyDecision,
    SlotValue,
)
from src.infrastructure.dst.hybrid_dst_adapter import HybridDSTAdapter
from src.infrastructure.nlg.template_nlg_adapter import TemplateNLGAdapter
from src.infrastructure.nlu.phobert_nlu_adapter import PhoBERTNLUAdapter
from src.infrastructure.policy.rule_policy_adapter import RulePolicyAdapter
from src.application.use_cases.tod_pipeline import TODPipelineUseCase


@pytest.fixture()
def pipeline() -> TODPipelineUseCase:
    return TODPipelineUseCase(
        nlu=PhoBERTNLUAdapter(),
        dst=HybridDSTAdapter(),
        policy=RulePolicyAdapter(),
        nlg=TemplateNLGAdapter(),
    )


class TestStateManagement:
    def test_creates_new_state(self, pipeline: TODPipelineUseCase) -> None:
        state = pipeline.get_or_create_state("s1")
        assert isinstance(state, DialogueState)
        assert state.session_id == "s1"

    def test_returns_existing_state(self, pipeline: TODPipelineUseCase) -> None:
        s1 = pipeline.get_or_create_state("s1")
        s1.slots["fromloc.city_name"] = "Hà Nội"
        s2 = pipeline.get_or_create_state("s1")
        assert s2.slots["fromloc.city_name"] == "Hà Nội"

    def test_clear_state(self, pipeline: TODPipelineUseCase) -> None:
        pipeline.get_or_create_state("s1")
        pipeline.clear_state("s1")
        state = pipeline.get_or_create_state("s1")
        assert state.turn_count == 0

    def test_clear_nonexistent_state_no_error(self, pipeline: TODPipelineUseCase) -> None:
        pipeline.clear_state("nonexistent")  # should not raise


class TestProcessTurn:
    async def test_greeting_turn(self, pipeline: TODPipelineUseCase) -> None:
        result = await pipeline.process_turn("s1", "xin chào")
        assert result["action"] == "greet"
        assert "Xin chào" in result["response_text"]
        assert result["nlu"]["intent"] == "greet"

    async def test_flight_booking_first_turn(self, pipeline: TODPipelineUseCase) -> None:
        result = await pipeline.process_turn("s1", "Tôi muốn bay đi Đà Nẵng")
        assert result["action"] in ("request_slot", "confirm")
        assert result["response_text"]
        assert result["nlu"]["intent"] == "atis_flight"

    async def test_multi_turn_booking(self, pipeline: TODPipelineUseCase) -> None:
        # Turn 1: destination only → should request origin
        r1 = await pipeline.process_turn("s1", "Tôi muốn đi Đà Nẵng")
        assert r1["action"] == "request_slot"

        # Turn 2: provide origin → both required slots filled
        r2 = await pipeline.process_turn("s1", "từ Hà Nội")
        # After filling both slots, could be confirm or execute depending on state
        assert r2["action"] in ("confirm", "execute")

    async def test_state_persists_across_turns(self, pipeline: TODPipelineUseCase) -> None:
        await pipeline.process_turn("s1", "đi Đà Nẵng")
        state = pipeline.get_or_create_state("s1")
        assert state.turn_count == 1
        assert state.slots["toloc.city_name"] is not None

    async def test_result_structure(self, pipeline: TODPipelineUseCase) -> None:
        result = await pipeline.process_turn("s1", "xin chào")
        assert "response_text" in result
        assert "nlu" in result
        assert "state" in result
        assert "action" in result
        assert "target_slot" in result
        assert "intent" in result["nlu"]
        assert "confidence" in result["nlu"]
        assert "slots" in result["nlu"]

    async def test_farewell_turn(self, pipeline: TODPipelineUseCase) -> None:
        result = await pipeline.process_turn("s1", "tạm biệt")
        assert result["action"] == "farewell"
        assert "Cảm ơn" in result["response_text"]

    async def test_independent_sessions(self, pipeline: TODPipelineUseCase) -> None:
        await pipeline.process_turn("s1", "đi Đà Nẵng")
        await pipeline.process_turn("s2", "xin chào")
        s1 = pipeline.get_or_create_state("s1")
        s2 = pipeline.get_or_create_state("s2")
        assert s1.slots["toloc.city_name"] is not None
        assert s2.intent == "greet"

    async def test_confirmation_flow(self, pipeline: TODPipelineUseCase) -> None:
        # Fill required slots
        await pipeline.process_turn("s1", "bay từ Hà Nội đến Đà Nẵng")
        r = await pipeline.process_turn("s1", "đúng rồi")
        # After affirm, should execute
        assert r["action"] == "execute"
