"""REST endpoint for TOD dialogue turns."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.interface.dtos.dialogue_dtos import (
    DialogueStateData,
    DialogueTurnRequest,
    DialogueTurnResponse,
    NLUData,
)

router = APIRouter(prefix="/api/v1/dialogue", tags=["dialogue"])


@router.post("/turn", response_model=DialogueTurnResponse)
async def dialogue_turn(
    body: DialogueTurnRequest,
    request: Request,
) -> DialogueTurnResponse:
    """Process a single dialogue turn through the TOD pipeline.

    Accepts user text, runs NLU -> DST -> Policy -> NLG,
    returns response text plus pipeline visualization data.
    """
    tod_pipeline = request.app.state.tod_pipeline

    result = await tod_pipeline.process_turn(body.session_id, body.text)

    state_data = result["state"]
    return DialogueTurnResponse(
        response_text=result["response_text"],
        nlu=NLUData(
            intent=result["nlu"]["intent"],
            confidence=result["nlu"]["confidence"],
            slots=result["nlu"]["slots"],
        ),
        state=DialogueStateData(
            session_id=state_data["session_id"],
            intent=state_data["intent"],
            intent_confidence=state_data["intent_confidence"],
            slots=state_data["slots"],
            confirmed=state_data["confirmed"],
            turn_count=state_data["turn_count"],
        ),
        action=result["action"],
        target_slot=result.get("target_slot"),
    )
