"""GenerateReminderUseCase — asynchronous post-call reminder generation.

After a call session ends, this use case:
  1. Fetches all *Message* records for the given *session_id* from PostgreSQL.
  2. Builds a commitment/follow-up detection prompt and calls the LLM with
     JSON output mode.
  3. Parses the LLM response into a list of
     :class:`~src.domain.entities.reminder.Reminder` entities.
     ``target_due_at`` is ``None`` when no parseable date is detected (FR-018).
  4. Persists each *Reminder* via :class:`PostgresReminderRepository`.
  5. Emits zero records when no actionable items are detected.

This use case is designed to run as a FastAPI ``BackgroundTask``; it MUST NOT
be awaited in the session teardown hot path.

Idempotency: if reminders already exist for *session_id*, the use case returns
the existing list without re-running extraction to avoid duplicate records.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.entities.reminder import Reminder
from src.domain.errors import PersistenceError

if TYPE_CHECKING:
    import uuid

    from src.application.ports.llm_port import LLMPort

logger = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an assistant that detects actionable follow-up commitments in a call \
transcript between a university call-center agent and a student.

Rules:
- Only extract commitments that are **explicitly stated** — a promise to follow
  up, call back, send information, or complete an action by a specific time.
- If a date/time is mentioned for a commitment, parse it as an ISO 8601
  datetime string (YYYY-MM-DDTHH:MM:SS). If no date/time is parseable, use null.
- NEVER guess, infer, or hallucinate values.
- If there are no actionable follow-up commitments, return an empty list.
- Respond with valid JSON — a JSON array (may be empty). No additional text.

Each item in the array must have:
{
  "description":   string,          // Short description of the commitment
  "target_due_at": "YYYY-MM-DDTHH:MM:SS" | null  // Parsed due date/time or null
}
"""


def _build_reminder_messages(
    transcript: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build the OpenAI-style message list for the commitment-detection call."""
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
                "Detect actionable follow-up commitments from the call transcript"
                " below:\n\n"
                f"---\n{transcript_text}\n---\n\n"
                "Respond with a JSON array only (may be empty)."
            ),
        },
    ]


def _parse_reminders_from_json(
    raw: Any,  # noqa: ANN401
    session_id: uuid.UUID,
) -> list[Reminder]:
    """Build a list of *Reminder* domain entities from the raw LLM JSON output.

    Invalid or missing ``description`` entries are skipped.
    Unparseable ``target_due_at`` values are set to ``None`` (FR-018).
    """
    if not isinstance(raw, list):
        logger.warning(
            "reminder_invalid_json_type",
            expected="list",
            got=type(raw).__name__,
            session_id=str(session_id),
        )
        return []

    reminders: list[Reminder] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        description: str | None = item.get("description") or None
        if not description:
            logger.warning(
                "reminder_missing_description",
                item=str(item)[:200],
                session_id=str(session_id),
            )
            continue

        raw_due = item.get("target_due_at")
        target_due_at: datetime | None = None
        if raw_due is not None:
            try:
                target_due_at = datetime.fromisoformat(str(raw_due))
            except (ValueError, TypeError):
                logger.warning(
                    "reminder_invalid_target_due_at",
                    value=raw_due,
                    session_id=str(session_id),
                )

        reminders.append(
            Reminder.create(
                session_id=session_id,
                description=description,
                target_due_at=target_due_at,
            )
        )

    return reminders


# ──────────────────────────────────────────────────────────────────────────────
# Use case
# ──────────────────────────────────────────────────────────────────────────────


class GenerateReminderUseCase:
    """Post-call reminder generation pipeline.

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

    async def execute(self, session_id: uuid.UUID) -> list[Reminder]:
        """Detect and persist *Reminder* records for *session_id*.

        Returns the list of persisted *Reminder* entities (may be empty when no
        actionable commitments were detected).

        Idempotent: if reminders already exist for *session_id*, returns the
        existing list without re-running extraction.

        This method is designed to run as a FastAPI ``BackgroundTask`` and
        MUST NOT be awaited in the session teardown hot path.
        """
        log = logger.bind(session_id=str(session_id))
        log.info("reminder_generation_started")

        # ── 1. Idempotency guard — return early if already processed ─────────
        try:
            async with self._session_factory() as db_session:
                from src.infrastructure.db.postgres.reminder_repo import (
                    PostgresReminderRepository,
                )

                reminder_repo = PostgresReminderRepository(db_session)
                existing = await reminder_repo.get_all_by_session_id(session_id)

            if existing:
                log.info(
                    "reminder_generation_skipped_already_complete",
                    count=len(existing),
                )
                return existing
        except Exception as exc:
            log.error("reminder_idempotency_check_failed", error=str(exc))
            # Proceed with generation — worst case we duplicate; acceptable for
            # background tasks.

        # ── 2. Fetch transcript ───────────────────────────────────────────────
        transcript: list[dict[str, str]] = []
        try:
            async with self._session_factory() as db_session:
                from src.infrastructure.db.postgres.call_session_repo import (
                    PostgresCallSessionRepository,
                )

                repo = PostgresCallSessionRepository(db_session)
                messages = await repo.list_messages_by_session(
                    session_id, limit=500
                )
                for msg in messages:
                    transcript.append(
                        {"role": str(msg.role), "content": msg.content}
                    )
        except Exception as exc:
            log.error("reminder_transcript_fetch_failed", error=str(exc))
            return []

        if not transcript:
            log.info("reminder_generation_empty_transcript")
            return []

        # ── 3. Call LLM in JSON-array mode ───────────────────────────────────
        extraction_messages = _build_reminder_messages(transcript)
        try:
            raw_json: Any = await self._llm.generate_json(
                extraction_messages,
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as exc:
            log.error("reminder_llm_extraction_failed", error=str(exc))
            return []

        # ── 4. Parse into Reminder entities ──────────────────────────────────
        try:
            reminders = _parse_reminders_from_json(raw_json, session_id)
        except Exception as exc:
            log.error("reminder_parse_failed", error=str(exc))
            return []

        if not reminders:
            log.info("reminder_generation_no_commitments_detected")
            return []

        # ── 5. Persist each Reminder ─────────────────────────────────────────
        persisted: list[Reminder] = []
        try:
            async with self._session_factory() as db_session:
                from src.infrastructure.db.postgres.reminder_repo import (
                    PostgresReminderRepository,
                )

                reminder_repo = PostgresReminderRepository(db_session)
                for reminder in reminders:
                    try:
                        saved = await reminder_repo.create(reminder)
                        persisted.append(saved)
                    except PersistenceError:
                        log.warning(
                            "reminder_single_persist_failed",
                            reminder_id=str(reminder.id),
                        )
                await db_session.commit()
        except Exception as exc:
            log.error("reminder_persist_failed", error=str(exc))
            return []

        log.info(
            "reminder_generation_complete",
            reminders_persisted=len(persisted),
        )
        return persisted
