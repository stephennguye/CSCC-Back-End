"""POST /api/v1/sessions — create a call session and issue a JWT.

Session creation flow:
  1. Validate optional request body.
  2. If no ``session_id`` provided: create a new *CallSession*.
  3. If ``session_id`` provided: re-issue a token for an existing active session.
  4. Issue a short-lived HS256 JWT scoped to the ``session_id``.
  5. Return :class:`~src.interface.dtos.rest_responses.SessionCreatedResponse`.

Rate limits (FR-024) are enforced by the Redis middleware in Phase 7 / T058.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from src.interface.dtos.rest_responses import SessionCreatedResponse

logger = structlog.get_logger(__name__)

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["sessions"])

# ── JWT configuration ─────────────────────────────────────────────────────────

JWT_SECRET: str = os.environ.get("JWT_SECRET", "changeme-use-a-real-secret-in-production")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_SECONDS: int = int(os.environ.get("SESSION_TOKEN_TTL_SECONDS", "300"))

# ── Request / Response models ─────────────────────────────────────────────────


class SessionCreateRequest(BaseModel):
    """Optional request body for POST /api/v1/sessions."""

    model_config = ConfigDict(strict=True, extra="forbid")

    session_id: uuid.UUID | None = None


# ── JWT helpers ───────────────────────────────────────────────────────────────


def _issue_jwt(session_id: str) -> str:
    """Issue an HS256 JWT scoped to *session_id*."""
    try:
        from jose import jwt  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "python-jose[cryptography] is required for JWT support"
        ) from exc

    now = int(datetime.now(UTC).timestamp())
    payload = {
        "sub": session_id,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> str:
    """Verify an HS256 JWT and return the ``sub`` claim (session_id).

    Raises:
        HTTPException 401: if the token is missing, expired, or invalid.
    """
    from jose import JWTError, jwt  # type: ignore[import-untyped]

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        session_id: str = payload.get("sub", "")
        if not session_id:
            raise JWTError("Missing sub claim")
        return session_id
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_TOKEN", "message": str(exc)},
        ) from exc


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ws_url(request: Request, session_id: str) -> str:
    """Build the WebSocket URL for the given session."""
    # Determine scheme
    scheme = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host", request.url.netloc)
    return f"{scheme}://{host}/ws/calls/{session_id}"


# ── Dependency injection via app.state ───────────────────────────────────────

from src.interface.dependencies import get_handle_call, get_session_factory


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/sessions",
    status_code=status.HTTP_201_CREATED,
    response_model=SessionCreatedResponse,
    response_model_by_alias=True,
    summary="Create a call session",
    responses={
        409: {"description": "Session already active"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal error"},
    },
)
async def create_session(
    request: Request,
    body: SessionCreateRequest | None = None,
    handle_call: Any = Depends(get_handle_call),  # noqa: ANN401, B008
) -> SessionCreatedResponse:
    """Create a *CallSession* and issue a short-lived Bearer JWT.

    Behaviour:
    - No body / empty body: create a brand-new ``CallSession`` and token.
    - Body with ``session_id``: re-issue a fresh token for an already-active
      session (used during WebSocket reconnection).  Returns 409 if the
      session does not exist or is not *active*.
    """
    request_body = body or SessionCreateRequest()

    if request_body.session_id is not None:
        # Token re-issuance path
        try:
            session = await handle_call.get_or_create_token_session(
                request_body.session_id
            )
        except Exception as exc:
            from src.domain.errors import SessionNotFoundError

            if isinstance(exc, SessionNotFoundError):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "code": "SESSION_NOT_ACTIVE",
                        "message": str(exc),
                    },
                ) from exc
            raise
    else:
        # New session creation path
        try:
            session = await handle_call.create_session()
        except Exception as exc:
            logger.exception("session_create_error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "SESSION_CREATE_FAILED", "message": str(exc)},
            ) from exc

    session_id_str = str(session.id)
    token = _issue_jwt(session_id_str)

    # Compute expires_at as a UTC datetime that the frontend can use directly
    expires_at = datetime.fromtimestamp(
        int(datetime.now(UTC).timestamp()) + TOKEN_TTL_SECONDS, tz=UTC
    )

    logger.info("session_created_response", session_id=session_id_str)

    return SessionCreatedResponse(
        session_id=session.id,
        token=token,
        expires_at=expires_at,
        ws_url=_ws_url(request, session_id_str),
    )


# ── GET /api/v1/sessions/{session_id} — Post-call summary ────────────────────


@router.get(
    "/sessions/{session_id}",
    summary="Retrieve post-call summary (transcript, claims, reminders)",
    responses={
        404: {"description": "Session not found"},
        400: {"description": "Malformed session_id UUID"},
    },
)
async def get_session_summary(
    session_id: str,
    session_factory: Any = Depends(get_session_factory),  # noqa: ANN401, B008
) -> dict:  # type: ignore[return]
    """Return a single post-call response aggregating transcript, claims and reminders.

    Response shape is designed to match the frontend ``PostCallResponseSchema``::

        {
            "session": {
                "id": "<uuid>",
                "startedAt": <epoch_ms>,
                "endedAt": <epoch_ms | null>,
                "outcome": "normal_end | error | abandoned"
            },
            "transcript": [
                {"index": 0, "speaker": "user|ai", "text": "...", "timestamp": <epoch_ms>}
            ],
            "claims": [
                {"index": 0, "text": "...", "speaker": "ai", "confidence": 0.95, "timestamp": <epoch_ms>}
            ],
            "reminders": [
                {"index": 0, "text": "...", "dueAt": "<iso-datetime> | null"}
            ]
        }
    """
    # ── Validate UUID ─────────────────────────────────────────────────────────
    try:
        parsed_id = uuid.UUID(session_id)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_UUID", "message": "Malformed session_id UUID"},
        ) from exc

    async with session_factory() as db_session:
        from src.infrastructure.db.postgres.call_session_repo import (
            PostgresCallSessionRepository,
        )
        from src.infrastructure.db.postgres.claim_repo import PostgresClaimRepository
        from src.infrastructure.db.postgres.reminder_repo import (
            PostgresReminderRepository,
        )

        session_repo = PostgresCallSessionRepository(db_session)
        claim_repo = PostgresClaimRepository(db_session)
        reminder_repo = PostgresReminderRepository(db_session)

        call_session = await session_repo.get_by_id(parsed_id)
        if call_session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "SESSION_NOT_FOUND", "message": f"Session {session_id} not found"},
            )

        messages = await session_repo.list_messages_by_session(parsed_id, limit=1000, offset=0)
        claim = await claim_repo.get_by_session_id(parsed_id)
        reminders = await reminder_repo.get_all_by_session_id(parsed_id)

    # ── Map session ───────────────────────────────────────────────────────────
    def _epoch_ms(dt: datetime | None) -> int | None:
        if dt is None:
            return None
        return int(dt.timestamp() * 1000)

    state_to_outcome = {"ended": "normal_end", "error": "error", "active": "abandoned"}
    state_val = call_session.state.value if hasattr(call_session.state, "value") else str(call_session.state)
    outcome = state_to_outcome.get(state_val, "abandoned")

    session_obj = {
        "id": str(call_session.id),
        "startedAt": _epoch_ms(call_session.created_at),
        "endedAt": _epoch_ms(call_session.ended_at),
        "outcome": outcome,
    }

    # ── Map transcript ────────────────────────────────────────────────────────
    transcript = [
        {
            "index": msg.sequence_number,
            "speaker": msg.role,       # "user" | "ai"
            "text": msg.content,
            "timestamp": _epoch_ms(msg.timestamp),
        }
        for msg in messages
    ]

    # ── Map claims ────────────────────────────────────────────────────────────
    claims_list: list[dict] = []
    if claim is not None and claim.schema_version != "not_extractable":
        claims_list = [
            {
                "index": 0,
                "text": claim.issue_category or claim.requested_action or "Extracted claim",
                "speaker": "ai",
                "confidence": float(claim.confidence) if claim.confidence is not None else 1.0,
                "timestamp": _epoch_ms(claim.extracted_at) or 0,
            }
        ]

    # ── Map reminders ─────────────────────────────────────────────────────────
    reminders_list = [
        {
            "index": idx,
            "text": r.description,
            "dueAt": r.target_due_at.isoformat() if r.target_due_at is not None else None,
        }
        for idx, r in enumerate(reminders)
    ]

    return {
        "session": session_obj,
        "transcript": transcript,
        "claims": claims_list,
        "reminders": reminders_list,
    }
