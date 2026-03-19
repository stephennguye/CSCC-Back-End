"""Tests for src.domain.entities.claim."""

from __future__ import annotations

import uuid
from datetime import date, datetime

import pytest

from src.domain.entities.claim import Claim
from src.domain.value_objects.session_state import UrgencyLevel


class TestClaimCreate:
    def test_create_minimal(self) -> None:
        sid = uuid.uuid4()
        claim = Claim.create(session_id=sid)
        assert isinstance(claim.id, uuid.UUID)
        assert claim.session_id == sid
        assert isinstance(claim.extracted_at, datetime)
        assert claim.schema_version == "v1"
        assert claim.student_name is None
        assert claim.issue_category is None
        assert claim.urgency_level is None
        assert claim.confidence is None
        assert claim.requested_action is None
        assert claim.follow_up_date is None

    def test_create_with_all_fields(self) -> None:
        sid = uuid.uuid4()
        cid = uuid.uuid4()
        ts = datetime(2026, 3, 18)
        fud = date(2026, 4, 1)
        claim = Claim.create(
            session_id=sid,
            claim_id=cid,
            extracted_at=ts,
            student_name="Nguyen Van A",
            issue_category="billing",
            urgency_level=UrgencyLevel.high,
            confidence=0.95,
            requested_action="refund",
            follow_up_date=fud,
            schema_version="v2",
        )
        assert claim.id == cid
        assert claim.session_id == sid
        assert claim.extracted_at == ts
        assert claim.student_name == "Nguyen Van A"
        assert claim.issue_category == "billing"
        assert claim.urgency_level == UrgencyLevel.high
        assert claim.confidence == 0.95
        assert claim.requested_action == "refund"
        assert claim.follow_up_date == fud
        assert claim.schema_version == "v2"


class TestClaimInvariants:
    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            Claim.create(session_id=uuid.uuid4(), confidence=1.5)

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            Claim.create(session_id=uuid.uuid4(), confidence=-0.1)

    def test_confidence_at_zero_passes(self) -> None:
        claim = Claim.create(session_id=uuid.uuid4(), confidence=0.0)
        assert claim.confidence == 0.0

    def test_confidence_at_one_passes(self) -> None:
        claim = Claim.create(session_id=uuid.uuid4(), confidence=1.0)
        assert claim.confidence == 1.0

    def test_confidence_none_passes(self) -> None:
        claim = Claim.create(session_id=uuid.uuid4())
        assert claim.confidence is None
