"""GET /api/v1/conversations/{session_id}/reminders — retrieve generated reminders.

Response behaviour (per contracts/rest-api.md):
  - Session exists and has ended (state = ended/error)
    → 200 with ``reminders_status: "complete"`` and the reminders array
      (may be empty when no actionable commitments were detected).
  - Session exists and is still active
    → 200 with ``reminders_status: "pending"`` and empty reminders array.
  - Session does not exist
    → 404 SESSION_NOT_FOUND.
  - Malformed session_id UUID
    → 400 INVALID_UUID.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from src.domain.value_objects.session_state import SessionState
from src.interface.dtos.rest_responses import ReminderData, RemindersResponse

logger = structlog.get_logger(__name__)

# ── Router ─────────────────────────────────────────────────────────────────────
router = APIRouter(tags=["conversations"])


# ── Dependency injection ───────────────────────────────────────────────────────

from src.interface.dependencies import get_session_factory

# ── Endpoint ───────────────────────────────────────────────────────────────────


@router.get(
    "/conversations/{session_id}/reminders",
    response_model=RemindersResponse,
    summary="Retrieve reminders generated from a session",
    responses={
        404: {"description": "Session not found"},
        400: {"description": "Malformed session_id UUID"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_reminders(
    session_id: str,
    session_factory: Any = Depends(get_session_factory),  # noqa: ANN401, B008
) -> RemindersResponse:
    """Return all reminders extracted from *session_id* after the call ended.

    Two possible response shapes (both HTTP 200):

    1. **Complete** — background worker has run; ``reminders`` array contains
       zero or more reminder objects.
    2. **Pending** — session is still active; background worker has not yet
       been enqueued.
    """
    # ── Validate UUID ─────────────────────────────────────────────────────────
    try:
        parsed_session_id = uuid.UUID(session_id)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_UUID", "message": "Malformed session_id UUID"},
        ) from exc

    log = logger.bind(session_id=session_id)

    async with session_factory() as db_session:
        from src.infrastructure.db.postgres.call_session_repo import (
            PostgresCallSessionRepository,
        )
        from src.infrastructure.db.postgres.reminder_repo import (
            PostgresReminderRepository,
        )

        session_repo = PostgresCallSessionRepository(db_session)
        reminder_repo = PostgresReminderRepository(db_session)

        # ── Verify session exists ─────────────────────────────────────────────
        call_session = await session_repo.get_by_id(parsed_session_id)
        if call_session is None:
            log.info("reminders_session_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": f"Session {session_id} not found",
                },
            )

        # ── Determine status based on session state ───────────────────────────
        # If the session is still active the background task has not yet been
        # enqueued, so we cannot have reminders yet.
        if call_session.state == SessionState.active:
            log.debug("reminders_pending_session_active")
            return RemindersResponse(
                session_id=parsed_session_id,
                reminders=[],
                total=0,
                reminders_status="pending",
            )

        # Session has ended — retrieve all reminders (may be empty list)
        reminders = await reminder_repo.get_all_by_session_id(parsed_session_id)

    # ── Build response ────────────────────────────────────────────────────────
    reminder_data = [
        ReminderData(
            id=r.id,
            description=r.description,
            target_due_at=r.target_due_at,
            created_at=r.created_at,
        )
        for r in reminders
    ]

    log.debug("reminders_found", count=len(reminder_data))
    return RemindersResponse(
        session_id=parsed_session_id,
        reminders=reminder_data,
        total=len(reminder_data),
        reminders_status="complete",
    )
