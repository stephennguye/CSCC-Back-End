"""Tests for src.domain.entities.call_session."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from src.domain.entities.call_session import CallSession
from src.domain.value_objects.session_state import SessionState


class TestCallSessionCreate:
    def test_create_default_uuid(self) -> None:
        session = CallSession.create()
        assert isinstance(session.id, uuid.UUID)
        assert session.state == SessionState.active
        assert isinstance(session.created_at, datetime)
        assert session.ended_at is None
        assert session.metadata is None

    def test_create_with_explicit_id(self) -> None:
        sid = uuid.uuid4()
        session = CallSession.create(session_id=sid)
        assert session.id == sid

    def test_create_with_metadata(self) -> None:
        meta = {"caller": "test-user", "channel": "web"}
        session = CallSession.create(metadata=meta)
        assert session.metadata == meta

    def test_create_independent_instances(self) -> None:
        s1 = CallSession.create()
        s2 = CallSession.create()
        assert s1.id != s2.id


class TestCallSessionStateTransitions:
    def test_end_sets_ended_state(self) -> None:
        session = CallSession.create()
        session.end()
        assert session.state == SessionState.ended
        assert session.ended_at is not None

    def test_end_with_explicit_timestamp(self) -> None:
        session = CallSession.create()
        ts = datetime(2026, 3, 18, 12, 0, 0)
        session.end(ended_at=ts)
        assert session.ended_at == ts

    def test_mark_error_sets_error_state(self) -> None:
        session = CallSession.create()
        session.mark_error()
        assert session.state == SessionState.error
        assert session.ended_at is not None

    def test_mark_error_with_explicit_timestamp(self) -> None:
        session = CallSession.create()
        ts = datetime(2026, 3, 18, 12, 0, 0)
        session.mark_error(ended_at=ts)
        assert session.ended_at == ts

    def test_cannot_end_already_ended_session(self) -> None:
        session = CallSession.create()
        session.end()
        with pytest.raises(ValueError, match="Cannot perform 'end'"):
            session.end()

    def test_cannot_end_error_session(self) -> None:
        session = CallSession.create()
        session.mark_error()
        with pytest.raises(ValueError, match="Cannot perform 'end'"):
            session.end()

    def test_cannot_mark_error_on_ended_session(self) -> None:
        session = CallSession.create()
        session.end()
        with pytest.raises(ValueError, match="Cannot perform 'mark_error'"):
            session.mark_error()


class TestCallSessionInvariants:
    def test_active_with_ended_at_raises(self) -> None:
        with pytest.raises(ValueError, match="ended_at must be null"):
            CallSession(
                id=uuid.uuid4(),
                state=SessionState.active,
                created_at=datetime.utcnow(),
                ended_at=datetime.utcnow(),
            )

    def test_ended_without_ended_at_raises(self) -> None:
        with pytest.raises(ValueError, match="ended_at must be set"):
            CallSession(
                id=uuid.uuid4(),
                state=SessionState.ended,
                created_at=datetime.utcnow(),
                ended_at=None,
            )

    def test_error_without_ended_at_raises(self) -> None:
        with pytest.raises(ValueError, match="ended_at must be set"):
            CallSession(
                id=uuid.uuid4(),
                state=SessionState.error,
                created_at=datetime.utcnow(),
                ended_at=None,
            )
