"""Contract tests for REST API request/response schemas.

Validates that all REST DTOs conform to their Pydantic schemas
and produce the expected JSON structure for the frontend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

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


class TestDialogueTurnRequestContract:
    def test_valid_from_json(self) -> None:
        req = DialogueTurnRequest.model_validate({
            "session_id": "abc-123",
            "text": "Tôi muốn đặt vé bay",
        })
        assert req.session_id == "abc-123"
        assert req.text == "Tôi muốn đặt vé bay"

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            DialogueTurnRequest.model_validate({"session_id": "s1", "text": ""})

    def test_rejects_text_over_1000(self) -> None:
        with pytest.raises(ValidationError):
            DialogueTurnRequest.model_validate({"session_id": "s1", "text": "a" * 1001})

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            DialogueTurnRequest.model_validate({})


class TestDialogueTurnResponseContract:
    def test_serialization_matches_frontend_schema(self) -> None:
        resp = DialogueTurnResponse(
            response_text="Xin chào! Tôi là trợ lý AI.",
            nlu=NLUData(
                intent="greet",
                confidence=0.95,
                slots={},
            ),
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
        data = resp.model_dump()
        # Frontend expects these exact keys
        assert set(data.keys()) == {
            "response_text", "nlu", "state", "action", "target_slot",
        }
        assert set(data["nlu"].keys()) == {"intent", "confidence", "slots"}
        assert set(data["state"].keys()) == {
            "session_id", "intent", "intent_confidence",
            "slots", "confirmed", "turn_count",
        }


class TestSessionCreatedResponseContract:
    def test_camelcase_aliases(self) -> None:
        resp = SessionCreatedResponse(
            session_id=uuid.uuid4(),
            token="jwt.token.here",
            expires_at=datetime.now(tz=UTC),
            ws_url="ws://localhost:8000/ws/calls/abc",
        )
        data = resp.model_dump(by_alias=True, mode="json")
        # Frontend expects camelCase keys
        assert "sessionId" in data
        assert "expiresAt" in data
        assert "wsUrl" in data
        assert "token" in data


class TestHealthResponseContract:
    def test_healthy_shape(self) -> None:
        resp = HealthResponse(
            status="healthy",
            timestamp=datetime.now(tz=UTC),
            services={
                "postgres": ServiceStatus(status="healthy", latency_ms=5),
                "redis": ServiceStatus(status="healthy", latency_ms=2),
                "openai": ServiceStatus(status="healthy", latency_ms=100),
                "huggingface": ServiceStatus(status="healthy", latency_ms=80),
                "google_cloud_stt": ServiceStatus(status="healthy", latency_ms=10),
                "tts": ServiceStatus(status="healthy", latency_ms=15),
            },
        )
        data = resp.model_dump(mode="json")
        assert data["status"] == "healthy"
        assert len(data["services"]) == 6
        for svc in data["services"].values():
            assert "status" in svc
            assert "latency_ms" in svc

    def test_degraded_with_error(self) -> None:
        resp = HealthResponse(
            status="degraded",
            timestamp=datetime.now(tz=UTC),
            services={
                "postgres": ServiceStatus(
                    status="unhealthy", latency_ms=0, error="connection refused"
                ),
            },
        )
        data = resp.model_dump(mode="json")
        assert data["status"] == "degraded"
        assert data["services"]["postgres"]["error"] == "connection refused"


class TestErrorResponseContract:
    def test_standard_error_envelope(self) -> None:
        resp = ErrorResponse(
            error=ErrorBody(
                code="SESSION_NOT_FOUND",
                message="Session abc-123 not found",
                request_id="req-456",
            ),
        )
        data = resp.model_dump(mode="json")
        assert set(data.keys()) == {"error"}
        assert set(data["error"].keys()) >= {"code", "message"}


class TestConversationHistoryContract:
    def test_response_shape(self) -> None:
        resp = ConversationHistoryResponse(
            session_id=uuid.uuid4(),
            state="ended",
            created_at=datetime.now(tz=UTC),
            ended_at=datetime.now(tz=UTC),
            messages=[
                MessageResponse(
                    id=uuid.uuid4(),
                    role="user",
                    content="Hello",
                    timestamp=datetime.now(tz=UTC),
                    sequence_number=1,
                ),
            ],
            total=1,
            limit=100,
            offset=0,
        )
        data = resp.model_dump(mode="json")
        assert data["state"] == "ended"
        assert len(data["messages"]) == 1
        msg = data["messages"][0]
        assert set(msg.keys()) >= {"id", "role", "content", "timestamp", "sequence_number"}


class TestClaimsResponseContract:
    def test_with_claim(self) -> None:
        resp = ClaimsResponse(
            session_id=uuid.uuid4(),
            claim=ClaimData(
                id=uuid.uuid4(),
                student_name="Test Student",
                issue_category="billing",
                urgency_level="high",
                confidence=0.9,
                extracted_at=datetime.now(tz=UTC),
                schema_version="v1",
            ),
        )
        data = resp.model_dump(mode="json")
        assert data["claim"] is not None
        assert data["claim"]["student_name"] == "Test Student"

    def test_without_claim_pending(self) -> None:
        resp = ClaimsResponse(
            session_id=uuid.uuid4(),
            claim=None,
            claim_status="pending",
        )
        data = resp.model_dump(mode="json")
        assert data["claim"] is None
        assert data["claim_status"] == "pending"


class TestRemindersResponseContract:
    def test_response_shape(self) -> None:
        resp = RemindersResponse(
            session_id=uuid.uuid4(),
            reminders=[
                ReminderData(
                    id=uuid.uuid4(),
                    description="Follow up on booking",
                    target_due_at=datetime.now(tz=UTC),
                    created_at=datetime.now(tz=UTC),
                ),
            ],
            total=1,
            reminders_status="complete",
        )
        data = resp.model_dump(mode="json")
        assert data["total"] == 1
        assert data["reminders_status"] == "complete"
        assert len(data["reminders"]) == 1


class TestPipelineStateMessageContract:
    def test_websocket_pipeline_state(self) -> None:
        msg = PipelineStateMessage(
            session_id="s1",
            stt_text="Tôi muốn bay",
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
            nlg_response="Bạn muốn bay từ đâu?",
        )
        data = msg.model_dump(mode="json")
        assert data["type"] == "pipeline_state"
        assert data["stt_text"] == "Tôi muốn bay"
