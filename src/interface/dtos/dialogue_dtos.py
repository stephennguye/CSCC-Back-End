"""DTOs for the dialogue turn REST endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DialogueTurnRequest(BaseModel):
    """Request body for POST /api/v1/dialogue/turn."""

    session_id: str = Field(..., description="Session identifier")
    text: str = Field(..., min_length=1, max_length=1000, description="User input text")


class SlotData(BaseModel):
    """A single slot name-value pair."""

    name: str
    value: str


class NLUData(BaseModel):
    """NLU output visualization data."""

    intent: str
    confidence: float
    slots: dict[str, str]


class DialogueStateData(BaseModel):
    """Serialized dialogue state."""

    session_id: str
    intent: str | None
    intent_confidence: float
    slots: dict[str, str | None]
    confirmed: bool
    turn_count: int


class DialogueTurnResponse(BaseModel):
    """Response body for POST /api/v1/dialogue/turn."""

    response_text: str
    nlu: NLUData
    state: DialogueStateData
    action: str
    target_slot: str | None = None


class PipelineStateMessage(BaseModel):
    """WebSocket message for pipeline visualization."""

    type: str = "pipeline_state"
    session_id: str
    stt_text: str | None = None
    nlu: NLUData
    state: DialogueStateData
    action: str
    target_slot: str | None = None
    nlg_response: str
