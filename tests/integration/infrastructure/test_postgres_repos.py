"""Integration tests for PostgreSQL repository implementations.

These tests run against the real Docker PostgreSQL instance (cscc_postgres).
Each test uses a transactional session that rolls back after completion.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from src.domain.entities.call_session import CallSession
from src.domain.entities.claim import Claim
from src.domain.entities.message import Message
from src.domain.entities.reminder import Reminder
from src.domain.errors import PersistenceError, SessionNotFoundError
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.session_state import SessionState, UrgencyLevel
from src.domain.value_objects.speaker_role import SpeakerRole
from src.infrastructure.db.postgres.call_session_repo import (
    PostgresCallSessionRepository,
)
from src.infrastructure.db.postgres.claim_repo import PostgresClaimRepository
from src.infrastructure.db.postgres.reminder_repo import PostgresReminderRepository


# ══════════════════════════════════════════════════════════════════════════════
# CallSessionRepository
# ══════════════════════════════════════════════════════════════════════════════


class TestCallSessionRepo:
    async def test_create_and_get(self, db_session, sample_call_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        created = await repo.create(sample_call_session)
        assert created.id == sample_call_session.id
        assert created.state == SessionState.active

        fetched = await repo.get_by_id(sample_call_session.id)
        assert fetched is not None
        assert fetched.id == sample_call_session.id
        assert fetched.state == SessionState.active

    async def test_get_nonexistent_returns_none(self, db_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_update_state_to_ended(self, db_session, sample_call_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        await repo.create(sample_call_session)

        updated = await repo.update_state(sample_call_session.id, SessionState.ended)
        assert updated.state == SessionState.ended
        assert updated.ended_at is not None

    async def test_update_state_to_error(self, db_session, sample_call_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        await repo.create(sample_call_session)

        updated = await repo.update_state(sample_call_session.id, SessionState.error)
        assert updated.state == SessionState.error
        assert updated.ended_at is not None

    async def test_update_state_nonexistent_raises(self, db_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        with pytest.raises(SessionNotFoundError):
            await repo.update_state(uuid.uuid4(), SessionState.ended)


class TestMessageRepo:
    async def test_append_and_list_messages(self, db_session, sample_call_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        await repo.create(sample_call_session)

        msg1 = Message.create(
            session_id=sample_call_session.id,
            role=SpeakerRole.user,
            content="Xin chào",
            sequence_number=1,
        )
        msg2 = Message.create(
            session_id=sample_call_session.id,
            role=SpeakerRole.ai,
            content="Xin chào! Tôi có thể giúp gì?",
            sequence_number=2,
        )

        saved1 = await repo.append_message(msg1)
        saved2 = await repo.append_message(msg2)
        assert saved1.sequence_number == 1
        assert saved2.sequence_number == 2

        messages = await repo.list_messages_by_session(sample_call_session.id)
        assert len(messages) == 2
        assert messages[0].content == "Xin chào"
        assert messages[1].role == SpeakerRole.ai

    async def test_count_messages(self, db_session, sample_call_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        await repo.create(sample_call_session)

        count = await repo.count_messages_by_session(sample_call_session.id)
        assert count == 0

        msg = Message.create(
            session_id=sample_call_session.id,
            role=SpeakerRole.user,
            content="Hello",
            sequence_number=1,
        )
        await repo.append_message(msg)

        count = await repo.count_messages_by_session(sample_call_session.id)
        assert count == 1

    async def test_list_messages_with_limit_offset(self, db_session, sample_call_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        await repo.create(sample_call_session)

        for i in range(5):
            msg = Message.create(
                session_id=sample_call_session.id,
                role=SpeakerRole.user,
                content=f"Message {i}",
                sequence_number=i + 1,
            )
            await repo.append_message(msg)

        page = await repo.list_messages_by_session(
            sample_call_session.id, limit=2, offset=1
        )
        assert len(page) == 2
        assert page[0].content == "Message 1"

    async def test_messages_for_nonexistent_session_empty(self, db_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        messages = await repo.list_messages_by_session(uuid.uuid4())
        assert messages == []

    async def test_message_with_confidence_score(self, db_session, sample_call_session) -> None:
        repo = PostgresCallSessionRepository(db_session)
        await repo.create(sample_call_session)

        msg = Message.create(
            session_id=sample_call_session.id,
            role=SpeakerRole.user,
            content="Tôi muốn bay",
            sequence_number=1,
            confidence_score=ConfidenceScore(value=0.92),
        )
        await repo.append_message(msg)

        messages = await repo.list_messages_by_session(sample_call_session.id)
        assert len(messages) == 1
        # Note: confidence roundtrip through DB may have float precision issues
        assert messages[0].confidence_score is not None
        assert abs(messages[0].confidence_score.value - 0.92) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# ClaimRepository
# ══════════════════════════════════════════════════════════════════════════════


class TestClaimRepo:
    async def test_create_and_get(self, db_session, sample_call_session) -> None:
        session_repo = PostgresCallSessionRepository(db_session)
        await session_repo.create(sample_call_session)

        claim_repo = PostgresClaimRepository(db_session)
        claim = Claim.create(
            session_id=sample_call_session.id,
            student_name="Nguyen Van A",
            issue_category="booking",
            urgency_level=UrgencyLevel.medium,
            confidence=0.85,
        )
        created = await claim_repo.create(claim)
        assert created.id == claim.id

        fetched = await claim_repo.get_by_session_id(sample_call_session.id)
        assert fetched is not None
        assert fetched.student_name == "Nguyen Van A"
        assert fetched.urgency_level == UrgencyLevel.medium
        assert abs(fetched.confidence - 0.85) < 0.01

    async def test_get_nonexistent_returns_none(self, db_session) -> None:
        repo = PostgresClaimRepository(db_session)
        result = await repo.get_by_session_id(uuid.uuid4())
        assert result is None

    async def test_upsert_creates_then_updates(self, db_session, sample_call_session) -> None:
        session_repo = PostgresCallSessionRepository(db_session)
        await session_repo.create(sample_call_session)

        claim_repo = PostgresClaimRepository(db_session)

        # First upsert creates
        claim1 = Claim.create(
            session_id=sample_call_session.id,
            issue_category="billing",
            confidence=0.7,
        )
        await claim_repo.upsert(claim1)

        fetched = await claim_repo.get_by_session_id(sample_call_session.id)
        assert fetched is not None
        assert fetched.issue_category == "billing"

        # Second upsert updates
        claim2 = Claim.create(
            session_id=sample_call_session.id,
            issue_category="refund",
            confidence=0.9,
        )
        await claim_repo.upsert(claim2)

        fetched2 = await claim_repo.get_by_session_id(sample_call_session.id)
        assert fetched2 is not None
        assert fetched2.issue_category == "refund"
        assert abs(fetched2.confidence - 0.9) < 0.01

    async def test_claim_with_null_fields(self, db_session, sample_call_session) -> None:
        session_repo = PostgresCallSessionRepository(db_session)
        await session_repo.create(sample_call_session)

        claim_repo = PostgresClaimRepository(db_session)
        claim = Claim.create(session_id=sample_call_session.id)
        await claim_repo.create(claim)

        fetched = await claim_repo.get_by_session_id(sample_call_session.id)
        assert fetched is not None
        assert fetched.student_name is None
        assert fetched.issue_category is None
        assert fetched.urgency_level is None
        assert fetched.confidence is None


# ══════════════════════════════════════════════════════════════════════════════
# ReminderRepository
# ══════════════════════════════════════════════════════════════════════════════


class TestReminderRepo:
    async def test_create_and_get(self, db_session, sample_call_session) -> None:
        session_repo = PostgresCallSessionRepository(db_session)
        await session_repo.create(sample_call_session)

        reminder_repo = PostgresReminderRepository(db_session)
        reminder = Reminder.create(
            session_id=sample_call_session.id,
            description="Follow up on flight booking",
            target_due_at=datetime(2026, 4, 1, 9, 0, tzinfo=UTC),
        )
        created = await reminder_repo.create(reminder)
        assert created.id == reminder.id

        fetched = await reminder_repo.get_all_by_session_id(sample_call_session.id)
        assert len(fetched) == 1
        assert fetched[0].description == "Follow up on flight booking"
        assert fetched[0].target_due_at is not None

    async def test_multiple_reminders_per_session(self, db_session, sample_call_session) -> None:
        session_repo = PostgresCallSessionRepository(db_session)
        await session_repo.create(sample_call_session)

        reminder_repo = PostgresReminderRepository(db_session)
        for i in range(3):
            r = Reminder.create(
                session_id=sample_call_session.id,
                description=f"Task {i + 1}",
            )
            await reminder_repo.create(r)

        fetched = await reminder_repo.get_all_by_session_id(sample_call_session.id)
        assert len(fetched) == 3

    async def test_get_nonexistent_session_returns_empty(self, db_session) -> None:
        repo = PostgresReminderRepository(db_session)
        result = await repo.get_all_by_session_id(uuid.uuid4())
        assert result == []

    async def test_reminder_without_due_date(self, db_session, sample_call_session) -> None:
        session_repo = PostgresCallSessionRepository(db_session)
        await session_repo.create(sample_call_session)

        reminder_repo = PostgresReminderRepository(db_session)
        reminder = Reminder.create(
            session_id=sample_call_session.id,
            description="No date reminder",
        )
        await reminder_repo.create(reminder)

        fetched = await reminder_repo.get_all_by_session_id(sample_call_session.id)
        assert len(fetched) == 1
        assert fetched[0].target_due_at is None
