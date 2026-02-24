"""GET /api/v1/conversations/{session_id}/claims — retrieve extracted claim.

Response behaviour (per contracts/rest-api.md):
  - Claim record present and schema_version != "not_extractable"
    → 200 with populated ClaimData (fields may be null when not determinable).
  - Claim record present with schema_version == "not_extractable"
    → 200 with ``{"claim": null, "claim_status": "not_extractable"}``.
  - No claim record (background worker not yet complete)
    → 200 with ``{"claim": null, "claim_status": "pending"}``.
  - Session does not exist
    → 404 SESSION_NOT_FOUND.
  - Malformed session_id UUID
    → 400 INVALID_UUID.

Security:
  - ``student_name`` is automatically decrypted by the pgcrypto TypeDecorator
    before the domain layer ever receives it; no additional decryption step is
    needed here.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.interface.dtos.rest_responses import ClaimData, ClaimsResponse

logger = structlog.get_logger(__name__)

# ── Router ──────────────────────────────────────────────────────────────────
router = APIRouter(tags=["conversations"])


# ── Dependency injection ────────────────────────────────────────────────────


def get_session_factory(request: Request) -> Any:  # noqa: ANN401
    """FastAPI dependency — retrieve the async session factory from app.state."""
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise RuntimeError("session_factory not initialised; app startup failed")
    return factory


# ── Endpoint ────────────────────────────────────────────────────────────────


@router.get(
    "/conversations/{session_id}/claims",
    response_model=ClaimsResponse,
    summary="Retrieve the structured claim extracted from a session",
    responses={
        404: {"description": "Session not found"},
        400: {"description": "Malformed session_id UUID"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_claims(
    session_id: str,
    session_factory: Any = Depends(get_session_factory),  # noqa: ANN401, B008
) -> ClaimsResponse:
    """Return the claim extracted from *session_id* after the call ended.

    Three possible responses (all HTTP 200):

    1. **Claim available** — claim object with extracted fields (may contain
       ``null`` values for fields that were not determinable from the transcript).
    2. **Pending** — background extraction worker has not yet completed.
    3. **Not extractable** — extraction ran but the transcript contained no
       claim-relevant information.
    """
    # ── Validate UUID ──────────────────────────────────────────────────────
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
        from src.infrastructure.db.postgres.claim_repo import (
            PostgresClaimRepository,
        )

        session_repo = PostgresCallSessionRepository(db_session)
        claim_repo = PostgresClaimRepository(db_session)

        # ── Verify session exists ──────────────────────────────────────────
        call_session = await session_repo.get_by_id(parsed_session_id)
        if call_session is None:
            log.info("claims_session_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": f"Session {session_id} not found",
                },
            )

        # ── Retrieve claim ─────────────────────────────────────────────────
        claim = await claim_repo.get_by_session_id(parsed_session_id)

    # ── Build response ─────────────────────────────────────────────────────
    if claim is None:
        # Background worker has not run yet
        log.debug("claims_pending")
        return ClaimsResponse(
            session_id=parsed_session_id,
            claim=None,
            claim_status="pending",
        )

    if claim.schema_version == "not_extractable":
        # Worker ran; transcript was empty or contained no extractable claim
        log.debug("claims_not_extractable")
        return ClaimsResponse(
            session_id=parsed_session_id,
            claim=None,
            claim_status="not_extractable",
        )

    # Claim record present — map to DTO (student_name already decrypted by ORM)
    urgency_str: str | None = (
        claim.urgency_level.value if claim.urgency_level is not None else None
    )

    claim_data = ClaimData(
        id=claim.id,
        student_name=claim.student_name,
        issue_category=claim.issue_category,
        urgency_level=urgency_str,  # type: ignore[arg-type]
        confidence=claim.confidence,
        requested_action=claim.requested_action,
        follow_up_date=claim.follow_up_date,
        extracted_at=claim.extracted_at,
        schema_version=claim.schema_version,
    )

    log.debug("claims_found", claim_id=str(claim.id))
    return ClaimsResponse(
        session_id=parsed_session_id,
        claim=claim_data,
    )
