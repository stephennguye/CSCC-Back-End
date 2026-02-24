"""Pydantic v2 models for all WebSocket frame types.

Inbound frames (Client → Server) and outbound frames (Server → Client) are
defined here.  All models use strict mode to reject unknown fields and enforce
type coercion rules.

See contracts/websocket.md for the authoritative schema definitions.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ════════════════════════════════════════════════════════════════════════════
# Shared envelope
# ════════════════════════════════════════════════════════════════════════════


class _WSFrame(BaseModel):
    """Base class for all WS text frames."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    session_id: uuid.UUID


# ════════════════════════════════════════════════════════════════════════════
# Inbound frames (Client → Server)
# ════════════════════════════════════════════════════════════════════════════

# ── audio.chunk ──────────────────────────────────────────────────────────────


class AudioChunkPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    sequence: int = Field(..., gt=0, description="Monotonically increasing chunk index")
    codec: Literal["pcm_16khz_mono", "opus_48khz"]
    data: str = Field(..., description="Base64-encoded raw audio bytes; max 4 KB decoded")


class AudioChunkFrame(_WSFrame):
    type: Literal["audio.chunk"]
    payload: AudioChunkPayload


# ── audio.end ────────────────────────────────────────────────────────────────


class AudioEndPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    sequence: int = Field(..., gt=0)


class AudioEndFrame(_WSFrame):
    type: Literal["audio.end"]
    payload: AudioEndPayload


# ── session.resume ───────────────────────────────────────────────────────────


class SessionResumePayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    last_sequence: int = Field(
        ...,
        ge=0,
        description="Last audio.chunk sequence the client successfully sent",
    )


class SessionResumeFrame(_WSFrame):
    type: Literal["session.resume"]
    payload: SessionResumePayload


# ── session.end ──────────────────────────────────────────────────────────────


class SessionEndPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SessionEndFrame(_WSFrame):
    type: Literal["session.end"]
    payload: SessionEndPayload = Field(default_factory=SessionEndPayload)


# ── Discriminated union of all valid inbound frames ──────────────────────────

InboundWSFrame = Annotated[
    AudioChunkFrame | AudioEndFrame | SessionResumeFrame | SessionEndFrame,
    Field(discriminator="type"),
]


# ════════════════════════════════════════════════════════════════════════════
# Outbound frames (Server → Client)
# ════════════════════════════════════════════════════════════════════════════

# ── transcript.partial ───────────────────────────────────────────────────────


class TranscriptPartialPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    segment_id: str


class TranscriptPartialFrame(_WSFrame):
    type: Literal["transcript.partial"] = "transcript.partial"
    payload: TranscriptPartialPayload


# ── transcript.final ─────────────────────────────────────────────────────────


class TranscriptFinalPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    segment_id: str


class TranscriptFinalFrame(_WSFrame):
    type: Literal["transcript.final"] = "transcript.final"
    payload: TranscriptFinalPayload


# ── transcript.low_confidence ────────────────────────────────────────────────


class TranscriptLowConfidencePayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    segment_id: str
    prompt_message: str


class TranscriptLowConfidenceFrame(_WSFrame):
    type: Literal["transcript.low_confidence"] = "transcript.low_confidence"
    payload: TranscriptLowConfidencePayload


# ── response.token ───────────────────────────────────────────────────────────


class ResponseTokenPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    token: str
    turn_id: str


class ResponseTokenFrame(_WSFrame):
    type: Literal["response.token"] = "response.token"
    payload: ResponseTokenPayload


# ── transcript.ai_final ──────────────────────────────────────────────────────


class TranscriptAIFinalPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    turn_id: str
    text: str
    timestamp: datetime
    sequence_number: int


class TranscriptAIFinalFrame(_WSFrame):
    type: Literal["transcript.ai_final"] = "transcript.ai_final"
    payload: TranscriptAIFinalPayload


# ── audio.response.start ─────────────────────────────────────────────────────


class AudioResponseStartPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    turn_id: str
    codec: Literal["pcm_16khz_mono", "opus_48khz"] = "pcm_16khz_mono"


class AudioResponseStartFrame(_WSFrame):
    type: Literal["audio.response.start"] = "audio.response.start"
    payload: AudioResponseStartPayload


# ── audio.response.end ───────────────────────────────────────────────────────


class AudioResponseEndPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    turn_id: str


class AudioResponseEndFrame(_WSFrame):
    type: Literal["audio.response.end"] = "audio.response.end"
    payload: AudioResponseEndPayload


# ── session.state ────────────────────────────────────────────────────────────

SessionStateValue = Literal[
    "connecting",
    "listening",
    "ai_thinking",
    "ai_speaking",
    "call_ended",
    "reconnecting",
    "error",
]


class SessionStatePayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    state: SessionStateValue
    previous_state: SessionStateValue | None = None
    timestamp: datetime


class SessionStateFrame(_WSFrame):
    type: Literal["session.state"] = "session.state"
    payload: SessionStatePayload


# ── error ────────────────────────────────────────────────────────────────────

WSErrorCode = Literal[
    "LLM_TIMEOUT",
    "LLM_FALLBACK_EXHAUSTED",
    "TRANSCRIPTION_ERROR",
    "PROMPT_INJECTION_DETECTED",
    "SESSION_ENDED",
    "SESSION_EXPIRED",
    "INVALID_PAYLOAD",
]


class ErrorPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    code: str  # WSErrorCode — kept as str for forward-compatibility
    message: str
    recoverable: bool


class ErrorFrame(_WSFrame):
    type: Literal["error"] = "error"
    payload: ErrorPayload


# ── barge_in.ack ─────────────────────────────────────────────────────────────


class BargeInAckPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    halted_turn_id: str


class BargeInAckFrame(_WSFrame):
    type: Literal["barge_in.ack"] = "barge_in.ack"
    payload: BargeInAckPayload


# ── llm.fallback ─────────────────────────────────────────────────────────────


class LLMFallbackPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    reason: str
    fallback_model: str


class LLMFallbackFrame(_WSFrame):
    type: Literal["llm.fallback"] = "llm.fallback"
    payload: LLMFallbackPayload


# ── rag.context ──────────────────────────────────────────────────────────────


class RAGSource(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    document_id: uuid.UUID
    chunk_index: int
    title: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class RAGContextPayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    turn_id: str
    sources: list[RAGSource]


class RAGContextFrame(_WSFrame):
    type: Literal["rag.context"] = "rag.context"
    payload: RAGContextPayload


# ── audio.response (JSON fallback) ───────────────────────────────────────────


class AudioResponsePayload(BaseModel):
    model_config = ConfigDict(strict=False, extra="forbid", frozen=True)

    sequence: int
    turn_id: str
    codec: Literal["pcm_16khz_mono", "opus_48khz"] = "pcm_16khz_mono"
    data: str  # base64-encoded


class AudioResponseFrame(_WSFrame):
    type: Literal["audio.response"] = "audio.response"
    payload: AudioResponsePayload


# ── Discriminated union of all outbound frames ───────────────────────────────

OutboundWSFrame = Annotated[
    TranscriptPartialFrame
    | TranscriptFinalFrame
    | TranscriptLowConfidenceFrame
    | ResponseTokenFrame
    | TranscriptAIFinalFrame
    | AudioResponseStartFrame
    | AudioResponseEndFrame
    | AudioResponseFrame
    | SessionStateFrame
    | ErrorFrame
    | BargeInAckFrame
    | LLMFallbackFrame
    | RAGContextFrame,
    Field(discriminator="type"),
]
