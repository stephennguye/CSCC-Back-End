"""ExtractClaimsUseCase — asynchronous post-call claim extraction.

After a call session ends, this use case:
  1. Fetches all *Message* records for the given *session_id* from PostgreSQL.
  2. Builds a schema-v1 extraction prompt and calls the LLM with JSON output
     mode (FR-014 structured / deterministic output).
  3. Parses the LLM response into a :class:`~src.domain.entities.claim.Claim`
     entity — any field the LLM cannot resolve MUST remain ``None`` (never
     guessed, per FR-013).
  4. Persists the *Claim* via an ``upsert`` for idempotency (safe to retry).

This use case is designed to run as a FastAPI ``BackgroundTask``; it MUST NOT
be awaited in the session teardown hot path.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.entities.claim import Claim
from src.domain.errors import ClaimExtractionError, PersistenceError

if TYPE_CHECKING:
    import uuid

    from src.application.ports.llm_port import LLMPort

logger = structlog.get_logger(__name__)

# ── Extraction prompt ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a structured data extraction assistant for a university call center.
Your task is to extract specific fields from a call transcript.

Rules:
- Only extract information that is **explicitly stated** in the transcript.
- If a field cannot be determined with confidence from the transcript, output null.
- NEVER guess, infer, or hallucinate values.
- Respond with valid JSON conforming to the schema below. No additional text.

JSON schema (schema_version: "v1"):
{
  "student_name":    string | null,   // Full name as stated by the caller
  "issue_category":  string | null,   // E.g. "enrollment", "fees", "grades", "housing", "other"
  "urgency_level":   "low" | "medium" | "high" | "critical" | null,
  "requested_action": string | null,  // Specific action requested by the caller
  "follow_up_date":  "YYYY-MM-DD" | null,  // ISO 8601 date if mentioned
  "confidence":      number | null    // Overall extraction confidence 0.0-1.0
}
"""


def _build_extraction_messages(
    transcript: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build the OpenAI-style message list for the extraction LLM call."""
    # Format transcript into a readable block
    lines = []
    for turn in transcript:
        role_label = "Caller" if turn.get("role") == "user" else "Agent"
        lines.append(f"{role_label}: {turn.get('content', '')}")
    transcript_text = "\n".join(lines) if lines else "(empty transcript)"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Extract structured claim data from the following call transcript:\n\n"
                f"---\n{transcript_text}\n---\n\n"
                "Respond with JSON only."
            ),
        },
    ]


def _parse_claim_from_json(
    raw: dict[str, Any],
    session_id: uuid.UUID,
) -> Claim:
    """Build a *Claim* domain entity from the raw LLM JSON output.

    Validates and coerces field types; unknown or invalid values are set to
    None rather than raising (null-not-guessed rule).
    """
    from src.domain.value_objects.session_state import UrgencyLevel

    student_name: str | None = raw.get("student_name") or None
    issue_category: str | None = raw.get("issue_category") or None
    requested_action: str | None = raw.get("requested_action") or None

    raw_urgency = raw.get("urgency_level")
    urgency_level: UrgencyLevel | None = None
    if raw_urgency is not None:
        try:
            urgency_level = UrgencyLevel(raw_urgency)
        except ValueError:
            logger.warning(
                "claim_invalid_urgency_level",
                value=raw_urgency,
                session_id=str(session_id),
            )

    raw_follow_up = raw.get("follow_up_date")
    follow_up_date: date | None = None
    if raw_follow_up is not None:
        try:
            follow_up_date = date.fromisoformat(str(raw_follow_up))
        except (ValueError, TypeError):
            logger.warning(
                "claim_invalid_follow_up_date",
                value=raw_follow_up,
                session_id=str(session_id),
            )

    raw_confidence = raw.get("confidence")
    confidence: float | None = None
    if raw_confidence is not None:
        try:
            c = float(raw_confidence)
            if 0.0 <= c <= 1.0:
                confidence = c
        except (ValueError, TypeError):
            logger.warning(
                "claim_invalid_confidence",
                value=raw_confidence,
                session_id=str(session_id),
            )

    return Claim.create(
        session_id=session_id,
        student_name=student_name,
        issue_category=issue_category,
        urgency_level=urgency_level,
        confidence=confidence,
        requested_action=requested_action,
        follow_up_date=follow_up_date,
        schema_version="v1",
        extracted_at=datetime.utcnow(),
    )


# ────────────────────────────────────────────────────────────────────────────
# Use case
# ────────────────────────────────────────────────────────────────────────────


class ExtractClaimsUseCase:
    """Post-call claim extraction pipeline.

    Parameters
    ----------
    llm:
        Primary LLM adapter (OpenAI or HuggingFace) used for JSON extraction.
    session_factory:
        Async session factory (``async_sessionmaker[AsyncSession]``).
    """

    def __init__(
        self,
        llm: LLMPort,
        session_factory: Any,  # noqa: ANN401
    ) -> None:
        self._llm = llm
        self._session_factory = session_factory

    async def execute(self, session_id: uuid.UUID) -> Claim | None:
        """Extract and persist the *Claim* for *session_id*.

        Returns the persisted *Claim*, or *None* when the transcript is empty
        or the LLM returns no extractable fields.

        This method is idempotent — calling it multiple times for the same
        *session_id* will upsert the record with the latest extraction result.

        Raises:
            ClaimExtractionError: only for unexpected errors that should be
                logged by the caller; extraction failures due to LLM response
                format issues are swallowed and result in a NULL-field Claim.
        """
        log = logger.bind(session_id=str(session_id))
        log.info("claim_extraction_started")

        # ── 1. Fetch transcript ──────────────────────────────────────────
        transcript: list[dict[str, str]] = []
        try:
            async with self._session_factory() as db_session:
                from src.infrastructure.db.postgres.call_session_repo import (
                    PostgresCallSessionRepository,
                )
                repo = PostgresCallSessionRepository(db_session)
                messages = await repo.list_messages_by_session(session_id, limit=500)
            for msg in messages:
                transcript.append({"role": str(msg.role), "content": msg.content})
        except Exception as exc:
            log.error("claim_transcript_fetch_failed", error=str(exc))
            raise ClaimExtractionError(
                f"Failed to fetch transcript for session {session_id}: {exc}"
            ) from exc

        if not transcript:
            # Persist a sentinel so the REST endpoint can return "not_extractable"
            # instead of "pending" — the task ran but there was nothing to extract.
            log.info("claim_extraction_empty_transcript_persisting_sentinel")
            sentinel = Claim.create(
                session_id=session_id,
                schema_version="not_extractable",
            )
            try:
                async with self._session_factory() as db_session:
                    from src.infrastructure.db.postgres.claim_repo import (
                        PostgresClaimRepository,
                    )
                    claim_repo = PostgresClaimRepository(db_session)
                    await claim_repo.upsert(sentinel)
                    await db_session.commit()
            except Exception as exc:
                log.warning("claim_sentinel_persist_failed", error=str(exc))
            return None

        # ── 2. Call LLM in JSON mode ─────────────────────────────────────
        extraction_messages = _build_extraction_messages(transcript)
        try:
            raw_json: dict[str, Any] = await self._llm.generate_json(
                extraction_messages,
                temperature=0.0,
                max_tokens=512,
            )
        except Exception as exc:
            log.error("claim_llm_extraction_failed", error=str(exc))
            raise ClaimExtractionError(
                f"LLM extraction failed for session {session_id}: {exc}"
            ) from exc

        # ── 3. Parse into Claim entity ───────────────────────────────────
        try:
            claim = _parse_claim_from_json(raw_json, session_id)
        except Exception as exc:
            log.error("claim_parse_failed", error=str(exc), raw=str(raw_json)[:500])
            raise ClaimExtractionError(
                f"Failed to parse LLM claim response for session {session_id}: {exc}"
            ) from exc

        # ── 4. Persist via upsert (idempotent) ───────────────────────────
        try:
            async with self._session_factory() as db_session:
                from src.infrastructure.db.postgres.claim_repo import (
                    PostgresClaimRepository,
                )
                claim_repo = PostgresClaimRepository(db_session)
                claim = await claim_repo.upsert(claim)
                await db_session.commit()
        except (ClaimExtractionError, PersistenceError):
            raise
        except Exception as exc:
            log.error("claim_persist_failed", error=str(exc))
            raise ClaimExtractionError(
                f"Failed to persist Claim for session {session_id}: {exc}"
            ) from exc

        log.info(
            "claim_extraction_complete",
            claim_id=str(claim.id),
            has_student_name=claim.student_name is not None,
            urgency_level=str(claim.urgency_level) if claim.urgency_level else None,
        )
        return claim
