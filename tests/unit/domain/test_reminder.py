"""Tests for src.domain.entities.reminder."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from src.domain.entities.reminder import Reminder


class TestReminderCreate:
    def test_create_minimal(self) -> None:
        sid = uuid.uuid4()
        reminder = Reminder.create(session_id=sid, description="Follow up on booking")
        assert isinstance(reminder.id, uuid.UUID)
        assert reminder.session_id == sid
        assert reminder.description == "Follow up on booking"
        assert isinstance(reminder.created_at, datetime)
        assert reminder.target_due_at is None

    def test_create_with_all_fields(self) -> None:
        sid = uuid.uuid4()
        rid = uuid.uuid4()
        ts = datetime(2026, 3, 18, 12, 0, 0)
        due = datetime(2026, 3, 25, 9, 0, 0)
        reminder = Reminder.create(
            session_id=sid,
            description="Confirm flight details",
            target_due_at=due,
            reminder_id=rid,
            created_at=ts,
        )
        assert reminder.id == rid
        assert reminder.created_at == ts
        assert reminder.target_due_at == due

    def test_multiple_reminders_per_session(self) -> None:
        sid = uuid.uuid4()
        r1 = Reminder.create(session_id=sid, description="Task 1")
        r2 = Reminder.create(session_id=sid, description="Task 2")
        assert r1.session_id == r2.session_id
        assert r1.id != r2.id


class TestReminderInvariants:
    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description must not be empty"):
            Reminder.create(session_id=uuid.uuid4(), description="")
