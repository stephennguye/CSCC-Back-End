"""Tests for exception handlers — domain error to HTTP mapping."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.domain.errors import (
    PayloadValidationError,
    PersistenceError,
    SessionAlreadyEndedError,
    SessionNotFoundError,
    TranscriptionError,
)


def _create_test_app():
    """Create a minimal FastAPI app with exception handlers for testing."""
    from fastapi import FastAPI

    from src.interface.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise/{error_type}")
    async def raise_error(error_type: str):
        error_map = {
            "session_not_found": SessionNotFoundError("Session XYZ not found"),
            "session_ended": SessionAlreadyEndedError("Session already ended"),
            "payload_validation": PayloadValidationError("Invalid audio codec"),
            "transcription": TranscriptionError("Whisper failed"),
            "persistence": PersistenceError("DB write failed"),
        }
        raise error_map[error_type]

    return app


@pytest.fixture()
def test_app():
    return _create_test_app()


class TestExceptionHandlerMapping:
    async def test_session_not_found_returns_404(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/raise/session_not_found")
            assert resp.status_code == 404
            assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"

    async def test_session_ended_returns_409(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/raise/session_ended")
            assert resp.status_code == 409
            assert resp.json()["error"]["code"] == "SESSION_ENDED"

    async def test_payload_validation_returns_400(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/raise/payload_validation")
            assert resp.status_code == 400
            assert resp.json()["error"]["code"] == "INVALID_PAYLOAD"

    async def test_transcription_error_returns_500(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/raise/transcription")
            assert resp.status_code == 500
            assert resp.json()["error"]["code"] == "TRANSCRIPTION_ERROR"

    async def test_persistence_returns_500(self, test_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/raise/persistence")
            assert resp.status_code == 500
            assert resp.json()["error"]["code"] == "PERSISTENCE_ERROR"
