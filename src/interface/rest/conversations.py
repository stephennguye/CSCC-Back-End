"""GET /api/v1/conversations/{session_id}/history — paginated transcript.

Response behaviour (per contracts/rest-api.md):
  - Session exists
    → 200 with ConversationHistoryResponse (messages may be empty for active session).
  - Session does not exist
    → 404 SESSION_NOT_FOUND.
  - Malformed session_id UUID
    → 400 INVALID_UUID.

Query parameters:
  - limit  (int, default 100): maximum messages to return.
  - offset (int, default 0): pagination offset.

Security:
  - Message.content is automatically decrypted by the pgcrypto TypeDecorator
    before reaching the domain layer; no additional decryption step is needed.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.interface.dtos.rest_responses import (
    ConversationHistoryResponse,
    MessageResponse,
)

logger = structlog.get_logger(__name__)

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(tags=["conversations"])


# ── Dependency injection ──────────────────────────────────────────────────────

from src.interface.dependencies import get_session_factory

# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.get(
    "/conversations/{session_id}/history",
    response_model=ConversationHistoryResponse,
    summary="Retrieve the ordered transcript for a session",
    responses={
        404: {"description": "Session not found"},
        400: {"description": "Malformed session_id UUID"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_conversation_history(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Max messages to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    session_factory: Any = Depends(get_session_factory),  # noqa: ANN401, B008
) -> ConversationHistoryResponse:
    """Return the ordered transcript for *session_id*.

    Supports pagination via ``limit`` and ``offset``.  The ``total`` field in
    the response reflects the full message count (before pagination), allowing
    clients to determine whether additional pages exist.
    """
    # ── Validate UUID ─────────────────────────────────────────────────────────
    try:
        parsed_session_id = uuid.UUID(session_id)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_UUID", "message": "Malformed session_id UUID"},
        ) from exc

    log = logger.bind(session_id=session_id, limit=limit, offset=offset)

    async with session_factory() as db_session:
        from src.infrastructure.db.postgres.call_session_repo import (
            PostgresCallSessionRepository,
        )

        repo = PostgresCallSessionRepository(db_session)

        # ── Verify session exists ─────────────────────────────────────────────
        call_session = await repo.get_by_id(parsed_session_id)
        if call_session is None:
            log.info("conversation_history_session_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": f"Session {session_id} not found",
                },
            )

        # ── Fetch total and paginated messages ────────────────────────────────
        total = await repo.count_messages_by_session(parsed_session_id)
        messages = await repo.list_messages_by_session(
            parsed_session_id, limit=limit, offset=offset
        )

    log.debug(
        "conversation_history_fetched",
        total=total,
        returned=len(messages),
    )

    # ── Build response ────────────────────────────────────────────────────────
    message_data = [
        MessageResponse(
            id=msg.id,
            role=msg.role.value,  # type: ignore[arg-type]
            content=msg.content,
            confidence_score=(
                msg.confidence_score.value if msg.confidence_score is not None else None
            ),
            timestamp=msg.timestamp,
            sequence_number=msg.sequence_number,
        )
        for msg in messages
    ]

    return ConversationHistoryResponse(
        session_id=parsed_session_id,
        state=call_session.state.value,  # type: ignore[arg-type]
        created_at=call_session.created_at,
        ended_at=call_session.ended_at,
        messages=message_data,
        total=total,
        limit=limit,
        offset=offset,
    )
