"""PostgreSQL implementation of ClaimRepository.

Persists :class:`~src.domain.entities.claim.Claim` records using
SQLAlchemy async ORM.

Uniqueness: at most one *Claim* per *session_id* (``uq_claim_session_id``).
``upsert`` is idempotent and safe to retry (FR-014 determinism guarantee).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.domain.entities.claim import Claim
from src.domain.errors import PersistenceError
from src.domain.value_objects.session_state import UrgencyLevel
from src.infrastructure.db.postgres.models import ClaimModel

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# ORM ↔ domain mapping helpers
# ────────────────────────────────────────────────────────────────────────────


def _model_to_claim(row: ClaimModel) -> Claim:
    return Claim(
        id=row.id,
        session_id=row.session_id,
        extracted_at=row.extracted_at,
        schema_version=row.schema_version,
        student_name=row.student_name,
        issue_category=row.issue_category,
        urgency_level=(
            UrgencyLevel(row.urgency_level)
            if row.urgency_level is not None
            else None
        ),
        confidence=row.confidence,
        requested_action=row.requested_action,
        follow_up_date=row.follow_up_date,
    )


# ────────────────────────────────────────────────────────────────────────────
# Repository
# ────────────────────────────────────────────────────────────────────────────


class PostgresClaimRepository:
    """Async PostgreSQL-backed :class:`ClaimRepository` implementation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # CRUD                                                                 #
    # ------------------------------------------------------------------ #

    async def create(self, claim: Claim) -> Claim:
        """Persist a new *Claim* and return it.

        Raises:
            PersistenceError: on database error (including unique-constraint
                violation when a Claim for *claim.session_id* already exists).
        """
        try:
            row = ClaimModel(
                id=claim.id,
                session_id=claim.session_id,
                student_name=claim.student_name,
                issue_category=claim.issue_category,
                urgency_level=(
                    claim.urgency_level.value
                    if claim.urgency_level is not None
                    else None
                ),
                confidence=claim.confidence,
                requested_action=claim.requested_action,
                follow_up_date=claim.follow_up_date,
                extracted_at=claim.extracted_at,
                schema_version=claim.schema_version,
            )
            self._session.add(row)
            await self._session.flush()
            logger.debug("claim_created", claim_id=str(claim.id), session_id=str(claim.session_id))
            return claim
        except Exception as exc:
            logger.error("claim_create_failed", session_id=str(claim.session_id), error=str(exc))
            raise PersistenceError(f"Failed to create Claim: {exc}") from exc

    async def upsert(self, claim: Claim) -> Claim:
        """Insert or replace the *Claim* for ``claim.session_id``.

        Uses PostgreSQL ``INSERT … ON CONFLICT (session_id) DO UPDATE`` so the
        operation is idempotent and safe to call multiple times (FR-014).
        """
        try:
            stmt = (
                pg_insert(ClaimModel)
                .values(
                    id=claim.id,
                    session_id=claim.session_id,
                    student_name=claim.student_name,
                    issue_category=claim.issue_category,
                    urgency_level=(
                        claim.urgency_level.value
                        if claim.urgency_level is not None
                        else None
                    ),
                    confidence=claim.confidence,
                    requested_action=claim.requested_action,
                    follow_up_date=claim.follow_up_date,
                    extracted_at=claim.extracted_at,
                    schema_version=claim.schema_version,
                )
                .on_conflict_do_update(
                    constraint="uq_claim_session_id",
                    set_={
                        "id": claim.id,
                        "student_name": claim.student_name,
                        "issue_category": claim.issue_category,
                        "urgency_level": (
                            claim.urgency_level.value
                            if claim.urgency_level is not None
                            else None
                        ),
                        "confidence": claim.confidence,
                        "requested_action": claim.requested_action,
                        "follow_up_date": claim.follow_up_date,
                        "extracted_at": claim.extracted_at,
                        "schema_version": claim.schema_version,
                    },
                )
            )
            await self._session.execute(stmt)
            await self._session.flush()
            logger.debug("claim_upserted", session_id=str(claim.session_id))
            return claim
        except Exception as exc:
            logger.error("claim_upsert_failed", session_id=str(claim.session_id), error=str(exc))
            raise PersistenceError(f"Failed to upsert Claim: {exc}") from exc

    async def get_by_session_id(self, session_id: uuid.UUID) -> Claim | None:
        """Return the *Claim* for *session_id*, or *None* when not yet extracted."""
        try:
            result = await self._session.execute(
                select(ClaimModel).where(ClaimModel.session_id == session_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _model_to_claim(row)
        except Exception as exc:
            logger.error("claim_get_failed", session_id=str(session_id), error=str(exc))
            raise PersistenceError(
                f"Failed to fetch Claim for session {session_id}: {exc}"
            ) from exc
