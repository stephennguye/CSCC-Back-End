"""Integration tests for REST API endpoints.

Uses httpx AsyncClient with the real FastAPI app (no mocks),
but overrides the lifespan to wire test-specific infrastructure.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from src.application.use_cases.tod_pipeline import TODPipelineUseCase
from src.infrastructure.dst.hybrid_dst_adapter import HybridDSTAdapter
from src.infrastructure.nlg.template_nlg_adapter import TemplateNLGAdapter
from src.infrastructure.nlu.phobert_nlu_adapter import PhoBERTNLUAdapter
from src.infrastructure.policy.rule_policy_adapter import RulePolicyAdapter


def _create_test_app() -> object:
    """Create a minimal FastAPI app with real TOD pipeline but no DB/Redis."""
    from fastapi import FastAPI

    from src.interface.api_router import register_routers
    from src.interface.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    register_routers(app)

    # Wire TOD pipeline directly on app.state (no lifespan needed)
    app.state.tod_pipeline = TODPipelineUseCase(
        nlu=PhoBERTNLUAdapter(),
        dst=HybridDSTAdapter(),
        policy=RulePolicyAdapter(),
        nlg=TemplateNLGAdapter(),
    )
    return app


@pytest.fixture()
def test_app():
    return _create_test_app()


class TestHealthEndpoint:
    async def test_health_returns_response(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/health")
            # 503 expected since no DB/Redis, but endpoint should respond
            assert resp.status_code in (200, 207, 503)
            body = resp.json()
            assert "status" in body
            assert "services" in body
            assert "timestamp" in body


class TestDialogueEndpoint:
    async def test_greet_turn(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": "test-session", "text": "xin chào"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["action"] == "greet"
            assert "Xin chào" in body["response_text"]
            assert body["nlu"]["intent"] == "greet"

    async def test_flight_booking_turn(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            sid = f"book-{uuid.uuid4().hex[:8]}"
            resp = await client.post(
                "/api/v1/dialogue/turn",
                json={
                    "session_id": sid,
                    "text": "đặt vé bay đi Đà Nẵng",
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["nlu"]["intent"] == "atis_flight"
            assert body["nlu"]["slots"].get("toloc.city_name") == "Đà Nẵng"
            assert body["state"]["turn_count"] == 1

    async def test_multi_turn_state_persistence(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            sid = f"multi-{uuid.uuid4().hex[:8]}"

            # Turn 1: provide destination
            r1 = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": sid, "text": "đi Đà Nẵng"},
            )
            assert r1.status_code == 200
            assert r1.json()["action"] == "request_slot"

            # Turn 2: provide origin
            r2 = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": sid, "text": "từ Hà Nội"},
            )
            assert r2.status_code == 200
            body2 = r2.json()
            assert body2["state"]["turn_count"] == 2
            assert body2["state"]["slots"]["fromloc.city_name"] == "Hà Nội"
            assert body2["state"]["slots"]["toloc.city_name"] == "Đà Nẵng"

    async def test_farewell_turn(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": "farewell-session", "text": "tạm biệt"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["action"] == "farewell"
            assert "Cảm ơn" in body["response_text"]

    async def test_empty_text_rejected(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": "s1", "text": ""},
            )
            assert resp.status_code == 400

    async def test_missing_session_id_rejected(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/dialogue/turn",
                json={"text": "xin chào"},
            )
            # 400 (custom validation handler) or 422 (FastAPI default)
            assert resp.status_code in (400, 422)

    async def test_response_structure(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": "s1", "text": "xin chào"},
            )
            body = resp.json()
            # Verify all expected fields are present
            assert "response_text" in body
            assert "nlu" in body
            assert "state" in body
            assert "action" in body
            assert "target_slot" in body
            # NLU subfields
            assert "intent" in body["nlu"]
            assert "confidence" in body["nlu"]
            assert "slots" in body["nlu"]
            # State subfields
            assert "session_id" in body["state"]
            assert "intent" in body["state"]
            assert "intent_confidence" in body["state"]
            assert "slots" in body["state"]
            assert "confirmed" in body["state"]
            assert "turn_count" in body["state"]

    async def test_full_booking_flow(self, test_app) -> None:
        """End-to-end: request slots -> confirm -> execute -> farewell."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            sid = f"flow-{uuid.uuid4().hex[:8]}"

            # Turn 1: destination only -> should request origin
            r = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": sid, "text": "Tôi muốn đi Đà Nẵng"},
            )
            assert r.json()["action"] == "request_slot"

            # Turn 2: provide origin -> should confirm
            r = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": sid, "text": "từ Hà Nội"},
            )
            action2 = r.json()["action"]
            assert action2 in ("confirm", "execute")

            # Turn 3: confirm -> should execute
            r = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": sid, "text": "đúng rồi"},
            )
            assert r.json()["action"] == "execute"

            # Turn 4: farewell
            r = await client.post(
                "/api/v1/dialogue/turn",
                json={"session_id": sid, "text": "tạm biệt"},
            )
            assert r.json()["action"] == "farewell"
