"""Domain error -> HTTP status code mapping.

Registers global exception handlers that translate domain errors
into structured JSON error envelopes.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.domain.errors import (
    PayloadValidationError,
    PersistenceError,
    SessionAlreadyEndedError,
    SessionNotFoundError,
    TranscriptionError,
)

_log = structlog.get_logger(__name__)


def _make_error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all domain-error-to-HTTP exception handlers on *app*."""

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(loc) for loc in e.get("loc", [])),
                "issue": e.get("msg", ""),
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "INVALID_PAYLOAD",
                    "message": "Request body validation failed.",
                    "details": details,
                }
            },
        )

    @app.exception_handler(SessionNotFoundError)
    async def handle_session_not_found(
        request: Request, exc: SessionNotFoundError
    ) -> JSONResponse:
        return _make_error_response("SESSION_NOT_FOUND", str(exc), status.HTTP_404_NOT_FOUND)

    @app.exception_handler(SessionAlreadyEndedError)
    async def handle_session_ended(
        request: Request, exc: SessionAlreadyEndedError
    ) -> JSONResponse:
        return _make_error_response(
            "SESSION_ENDED", str(exc), status.HTTP_409_CONFLICT
        )

    @app.exception_handler(PayloadValidationError)
    async def handle_payload_validation(
        request: Request, exc: PayloadValidationError
    ) -> JSONResponse:
        return _make_error_response(
            "INVALID_PAYLOAD", str(exc), status.HTTP_400_BAD_REQUEST
        )

    @app.exception_handler(TranscriptionError)
    async def handle_transcription_error(
        request: Request, exc: TranscriptionError
    ) -> JSONResponse:
        return _make_error_response(
            "TRANSCRIPTION_ERROR", str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    @app.exception_handler(PersistenceError)
    async def handle_persistence(
        request: Request, exc: PersistenceError
    ) -> JSONResponse:
        _log.error("persistence_error", detail=str(exc))
        return _make_error_response(
            "PERSISTENCE_ERROR",
            "A storage operation failed. Please try again.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
