from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""

class TranscriptionError(DomainError):
    """faster-whisper failed or returned invalid output."""

class LLMTimeoutError(DomainError):
    """OpenAI / HuggingFace call exceeded the configured timeout."""

class LLMFallbackError(DomainError):
    """The LLM primary provider is unavailable; fallback was triggered."""

class LLMFallbackExhaustedError(DomainError):
    """Both primary and fallback LLM providers are unavailable."""

class RAGGroundingError(DomainError):
    """Retrieval returned no usable context above the confidence threshold."""

class PromptInjectionDetectedError(DomainError):
    """The prompt sanitizer blocked a detected instruction-override pattern."""

class SessionNotFoundError(DomainError):
    """Session ID does not exist in PostgreSQL."""

class SessionAlreadyEndedError(DomainError):
    """Operation attempted on a session in *ended* or *error* state."""

class ClaimExtractionError(DomainError):
    """Background worker failed to produce a valid claim schema."""

class PersistenceError(DomainError):
    """A database read/write operation failed."""

class PayloadValidationError(DomainError):
    """An inbound WS frame or REST payload failed schema validation."""
