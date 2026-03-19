"""Tests for src.domain.errors — error class hierarchy."""

from __future__ import annotations

from src.domain.errors import (
    DomainError,
    PayloadValidationError,
    PersistenceError,
    SessionAlreadyEndedError,
    SessionNotFoundError,
    TranscriptionError,
)


class TestErrorHierarchy:
    def test_all_inherit_from_domain_error(self) -> None:
        errors = [
            TranscriptionError,
            SessionNotFoundError,
            SessionAlreadyEndedError,
            PersistenceError,
            PayloadValidationError,
        ]
        for err_cls in errors:
            assert issubclass(err_cls, DomainError), f"{err_cls.__name__} is not a DomainError"

    def test_domain_error_inherits_exception(self) -> None:
        assert issubclass(DomainError, Exception)

    def test_error_messages_preserved(self) -> None:
        msg = "Session abc not found"
        err = SessionNotFoundError(msg)
        assert str(err) == msg

    def test_error_can_be_raised_and_caught(self) -> None:
        try:
            raise TranscriptionError("STT failed")
        except DomainError as e:
            assert str(e) == "STT failed"
