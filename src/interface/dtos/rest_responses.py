"""Pydantic v2 REST response DTO models.

All response schemas are defined here including the standard error envelope.
See contracts/rest-api.md for the authoritative schema definitions.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# ════════════════════════════════════════════════════════════════════════════
# Common / shared
# ════════════════════════════════════════════════════════════════════════════


class _BaseResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# ── Standard error envelope ──────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    field: str
    issue: str


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ════════════════════════════════════════════════════════════════════════════
# POST /sessions
# ════════════════════════════════════════════════════════════════════════════


class SessionCreatedResponse(_BaseResponse):
    """Response body for POST /api/v1/sessions (201 Created).

    Serialised with ``by_alias=True`` so the frontend receives camelCase:
    ``sessionId``, ``expiresAt``, ``wsUrl``.
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    session_id: uuid.UUID
    token: str = Field(..., description="Short-lived JWT scoped to this session")
    expires_at: datetime = Field(..., description="UTC datetime when the token expires")
    ws_url: str = Field(..., description="Pre-formed WebSocket URL for this session")


# ════════════════════════════════════════════════════════════════════════════
# GET /conversations/{session_id}/history
# ════════════════════════════════════════════════════════════════════════════


class MessageResponse(_BaseResponse):
    id: uuid.UUID
    role: Literal["user", "ai"]
    content: str
    confidence_score: float | None = None
    timestamp: datetime
    sequence_number: int


class ConversationHistoryResponse(_BaseResponse):
    """Response body for GET /api/v1/conversations/{session_id}/history."""

    session_id: uuid.UUID
    state: Literal["active", "ended", "error"]
    created_at: datetime
    ended_at: datetime | None = None
    messages: list[MessageResponse]
    total: int
    limit: int
    offset: int


# ════════════════════════════════════════════════════════════════════════════
# GET /conversations/{session_id}/claims
# ════════════════════════════════════════════════════════════════════════════


class ClaimData(_BaseResponse):
    id: uuid.UUID
    student_name: str | None = None
    issue_category: str | None = None
    urgency_level: Literal["low", "medium", "high", "critical"] | None = None
    confidence: float | None = None
    requested_action: str | None = None
    follow_up_date: date | None = None
    extracted_at: datetime
    schema_version: str


class ClaimsResponse(_BaseResponse):
    """Response body for GET /api/v1/conversations/{session_id}/claims."""

    session_id: uuid.UUID
    claim: ClaimData | None = None
    claim_status: Literal["pending", "not_extractable"] | None = Field(
        default=None,
        description="Present only when claim is null",
    )


# ════════════════════════════════════════════════════════════════════════════
# GET /conversations/{session_id}/reminders
# ════════════════════════════════════════════════════════════════════════════


class ReminderData(_BaseResponse):
    id: uuid.UUID
    description: str
    target_due_at: datetime | None = None
    created_at: datetime


class RemindersResponse(_BaseResponse):
    """Response body for GET /api/v1/conversations/{session_id}/reminders."""

    session_id: uuid.UUID
    reminders: list[ReminderData]
    total: int
    reminders_status: Literal["pending", "complete"]


# ════════════════════════════════════════════════════════════════════════════
# POST /documents/ingest
# ════════════════════════════════════════════════════════════════════════════


class DocumentIngestRequest(_BaseResponse):
    """Request body for POST /api/v1/documents/ingest."""

    content: str = Field(..., description="Raw extracted text; max 2 MB")
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = None


class DocumentIngestResponse(_BaseResponse):
    """Response body for POST /api/v1/documents/ingest (202 Accepted)."""

    document_id: uuid.UUID
    status: Literal["ingesting"]
    message: str


# ════════════════════════════════════════════════════════════════════════════
# GET /health
# ════════════════════════════════════════════════════════════════════════════


class ServiceStatus(_BaseResponse):
    status: Literal["healthy", "unhealthy"]
    latency_ms: int | None = None
    error: str | None = None


class HealthResponse(_BaseResponse):
    """Response body for GET /api/v1/health."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    services: dict[str, ServiceStatus]
