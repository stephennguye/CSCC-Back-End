"""Tests for src.domain.entities.message."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from src.domain.entities.message import Message
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.speaker_role import SpeakerRole


class TestMessageCreate:
    def test_create_minimal(self) -> None:
        sid = uuid.uuid4()
        msg = Message.create(
            session_id=sid,
            role=SpeakerRole.user,
            content="Xin chào",
            sequence_number=1,
        )
        assert isinstance(msg.id, uuid.UUID)
        assert msg.session_id == sid
        assert msg.role == SpeakerRole.user
        assert msg.content == "Xin chào"
        assert msg.sequence_number == 1
        assert isinstance(msg.timestamp, datetime)
        assert msg.confidence_score is None

    def test_create_with_all_fields(self) -> None:
        sid = uuid.uuid4()
        mid = uuid.uuid4()
        ts = datetime(2026, 3, 18, 12, 0, 0)
        cs = ConfidenceScore(value=0.95)
        msg = Message.create(
            session_id=sid,
            role=SpeakerRole.ai,
            content="Tôi có thể giúp gì?",
            sequence_number=2,
            confidence_score=cs,
            timestamp=ts,
            message_id=mid,
        )
        assert msg.id == mid
        assert msg.role == SpeakerRole.ai
        assert msg.timestamp == ts
        assert msg.confidence_score == cs

    def test_ai_role(self) -> None:
        msg = Message.create(
            session_id=uuid.uuid4(),
            role=SpeakerRole.ai,
            content="Response text",
            sequence_number=1,
        )
        assert msg.role == SpeakerRole.ai
        assert msg.role.value == "ai"


class TestMessageInvariants:
    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError, match="content must not be empty"):
            Message.create(
                session_id=uuid.uuid4(),
                role=SpeakerRole.user,
                content="",
                sequence_number=1,
            )

    def test_zero_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence_number must be a positive"):
            Message.create(
                session_id=uuid.uuid4(),
                role=SpeakerRole.user,
                content="Hello",
                sequence_number=0,
            )

    def test_negative_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence_number must be a positive"):
            Message.create(
                session_id=uuid.uuid4(),
                role=SpeakerRole.user,
                content="Hello",
                sequence_number=-1,
            )
