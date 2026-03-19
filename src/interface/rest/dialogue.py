"""REST endpoint for TOD dialogue turns."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends

from src.interface.dependencies import get_tod_pipeline
from src.interface.dtos.dialogue_dtos import (
    DialogueStateData,
    DialogueTurnRequest,
    DialogueTurnResponse,
    NLUData,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/dialogue", tags=["dialogue"])


@router.post("/turn", response_model=DialogueTurnResponse)
async def dialogue_turn(
    body: DialogueTurnRequest,
    tod_pipeline: Any = Depends(get_tod_pipeline),  # noqa: ANN401, B008
) -> DialogueTurnResponse:
    """Process a single dialogue turn through the TOD pipeline.

    Accepts user text, runs NLU -> DST -> Policy -> NLG,
    returns response text plus pipeline visualization data.
    """
    log = logger.bind(session_id=body.session_id)
    log.info("dialogue_turn_received", text_length=len(body.text))

    result = await tod_pipeline.process_turn(body.session_id, body.text)

    log.info(
        "dialogue_turn_complete",
        intent=result["nlu"]["intent"],
        action=result["action"],
    )

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
