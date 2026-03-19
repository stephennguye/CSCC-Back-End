from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""

class TranscriptionError(DomainError):
    """faster-whisper failed or returned invalid output."""

class TTSSynthesisError(DomainError):
    """TTS engine failed to synthesize audio."""

class SessionNotFoundError(DomainError):
    """Session ID does not exist in PostgreSQL."""

class SessionAlreadyEndedError(DomainError):
    """Operation attempted on a session in *ended* or *error* state."""

class PersistenceError(DomainError):
    """A database read/write operation failed."""

class PayloadValidationError(DomainError):
    """An inbound WS frame or REST payload failed schema validation."""
