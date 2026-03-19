"""Tests for interface DTOs — dialogue DTOs and REST response models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.interface.dtos.dialogue_dtos import (
    DialogueStateData,
    DialogueTurnRequest,
    DialogueTurnResponse,
    NLUData,
    PipelineStateMessage,
)
from src.interface.dtos.rest_responses import (
    ClaimData,
    ClaimsResponse,
    ConversationHistoryResponse,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    ReminderData,
    RemindersResponse,
    ServiceStatus,
    SessionCreatedResponse,
)


class TestDialogueTurnRequest:
    def test_valid_request(self) -> None:
        req = DialogueTurnRequest(session_id="s1", text="xin chào")
        assert req.session_id == "s1"
        assert req.text == "xin chào"

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DialogueTurnRequest(session_id="s1", text="")

    def test_max_length_text(self) -> None:
        req = DialogueTurnRequest(session_id="s1", text="a" * 1000)
        assert len(req.text) == 1000

    def test_over_max_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DialogueTurnRequest(session_id="s1", text="a" * 1001)


class TestNLUData:
    def test_valid(self) -> None:
        nlu = NLUData(intent="atis_flight", confidence=0.92, slots={"fromloc.city_name": "Hà Nội"})
        assert nlu.intent == "atis_flight"
        assert nlu.confidence == 0.92
        assert nlu.slots["fromloc.city_name"] == "Hà Nội"


class TestDialogueTurnResponse:
    def test_full_response(self) -> None:
        resp = DialogueTurnResponse(
            response_text="Xin chào!",
            nlu=NLUData(intent="greet", confidence=0.95, slots={}),
            state=DialogueStateData(
                session_id="s1",
                intent="greet",
                intent_confidence=0.95,
                slots={},
                confirmed=False,
                turn_count=1,
            ),
            action="greet",
            target_slot=None,
        )
        assert resp.response_text == "Xin chào!"
        assert resp.action == "greet"


class TestPipelineStateMessage:
    def test_pipeline_state_type(self) -> None:
        msg = PipelineStateMessage(
            session_id="s1",
            nlu=NLUData(intent="atis_flight", confidence=0.9, slots={}),
            state=DialogueStateData(
                session_id="s1",
                intent="atis_flight",
                intent_confidence=0.9,
                slots={},
                confirmed=False,
                turn_count=1,
            ),
            action="request_slot",
            target_slot="fromloc.city_name",
            nlg_response="Bạn bay từ đâu?",
        )
        assert msg.type == "pipeline_state"


class TestSessionCreatedResponse:
    def test_camel_case_serialization(self) -> None:
        sid = uuid.uuid4()
        resp = SessionCreatedResponse(
            session_id=sid,
            token="jwt-token",
            expires_at=datetime.now(tz=UTC),
            ws_url="ws://localhost:8000/ws/calls/123",
        )
        data = resp.model_dump(by_alias=True)
        assert "sessionId" in data
        assert "expiresAt" in data
        assert "wsUrl" in data


class TestHealthResponse:
    def test_healthy(self) -> None:
        resp = HealthResponse(
            status="healthy",
            timestamp=datetime.now(tz=UTC),
            services={
                "postgres": ServiceStatus(status="healthy", latency_ms=5),
                "redis": ServiceStatus(status="healthy", latency_ms=2),
            },
        )
        assert resp.status == "healthy"
        assert resp.services["postgres"].status == "healthy"

    def test_degraded_with_error(self) -> None:
        resp = HealthResponse(
            status="degraded",
            timestamp=datetime.now(tz=UTC),
            services={
                "postgres": ServiceStatus(status="unhealthy", latency_ms=0, error="connection refused"),
            },
        )
        assert resp.services["postgres"].error == "connection refused"


class TestErrorResponse:
    def test_error_envelope(self) -> None:
        resp = ErrorResponse(
            error=ErrorBody(code="SESSION_NOT_FOUND", message="Session not found")
        )
        assert resp.error.code == "SESSION_NOT_FOUND"


class TestMessageResponse:
    def test_valid(self) -> None:
        msg = MessageResponse(
            id=uuid.uuid4(),
            role="user",
            content="Hello",
            timestamp=datetime.now(tz=UTC),
            sequence_number=1,
        )
        assert msg.role == "user"


class TestClaimsResponse:
    def test_with_claim(self) -> None:
        resp = ClaimsResponse(
            session_id=uuid.uuid4(),
            claim=ClaimData(
                id=uuid.uuid4(),
                extracted_at=datetime.now(tz=UTC),
                schema_version="v1",
                student_name="Test",
            ),
        )
        assert resp.claim is not None
        assert resp.claim.student_name == "Test"

    def test_without_claim(self) -> None:
        resp = ClaimsResponse(
            session_id=uuid.uuid4(),
            claim=None,
            claim_status="pending",
        )
        assert resp.claim is None
        assert resp.claim_status == "pending"


class TestRemindersResponse:
    def test_with_reminders(self) -> None:
        resp = RemindersResponse(
            session_id=uuid.uuid4(),
            reminders=[
                ReminderData(
                    id=uuid.uuid4(),
                    description="Follow up",
                    created_at=datetime.now(tz=UTC),
                ),
            ],
            total=1,
            reminders_status="complete",
        )
        assert resp.total == 1
        assert len(resp.reminders) == 1
