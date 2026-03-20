"""WebSocket call controller.

Endpoint: ``ws://{host}/ws/calls/{session_id}``

Lifecycle::

    Client ──upgrade(Bearer token)──→ validate JWT & session
           ──audio.chunk frames─────→ queue audio chunks
           ──audio.end──────────────→ trigger STT→TOD→TTS pipeline
           ──session.end────────────→ graceful teardown
           ↩ disconnect (abrupt)    → same teardown path as session.end
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from src.domain.errors import PayloadValidationError
from src.interface.dtos.ws_messages import InboundWSFrame
from src.interface.validators.audio_frame import validate_audio_chunk

if TYPE_CHECKING:
    from src.application.use_cases.handle_call import HandleCallUseCase

logger = structlog.get_logger(__name__)

router = APIRouter()

# ── Dependency injection via app.state ───────────────────────────────────────

from src.interface.dependencies import get_handle_call_ws as get_handle_call

# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_token(websocket: WebSocket) -> str | None:
    """Extract Bearer token from the Authorization header or ``token`` query param."""
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # Fallback: query parameter (some WS clients cannot set custom headers)
    return websocket.query_params.get("token")


async def _send_text(websocket: WebSocket, frame: dict[str, Any]) -> None:
    """Serialize *frame* to JSON and send as a text WebSocket message."""
    try:
        await websocket.send_text(json.dumps(frame, default=str))
    except Exception as exc:
        logger.warning("ws_send_text_error", frame_type=frame.get("type"), error=str(exc))


async def _send_binary(websocket: WebSocket, data: bytes) -> None:
    """Send raw audio bytes as a binary WebSocket message."""
    try:
        await websocket.send_bytes(data)
    except Exception as exc:
        logger.warning("ws_send_binary_error", error=str(exc), data_len=len(data))


# ── State value mapping: BE internal → FE expected ————————————————————————————

_STATE_MAP: dict[str, str] = {
    "connecting": "connecting",
    "listening": "listening",
    "ai_thinking": "thinking",
    "ai_speaking": "speaking",
    "call_ended": "ended",
    "reconnecting": "reconnecting",
    "error": "error",
}


def _adapt_outbound_frame(frame: dict[str, Any]) -> dict[str, Any] | None:
    """Translate the BE internal WS frame format into the FE-expected flat format.

    BE uses dot-notation types with nested ``payload`` dicts.
    FE expects underscore-types with flat top-level fields.
    Returns ``None`` for frames that have no FE equivalent (they are dropped).
    """
    import time as _time

    frame_type = frame.get("type", "")
    payload: dict[str, Any] = frame.get("payload", {})
    ts_ms: int = int(_time.time() * 1000)

    if frame_type == "session.state":
        raw_state = payload.get("state", "")
        return {
            "type": "call_state",
            "state": _STATE_MAP.get(raw_state, raw_state),
            "timestamp": ts_ms,
        }

    if frame_type == "transcript.partial":
        # Streaming user ASR token
        return {
            "type": "transcript_token",
            "speaker": "user",
            "token": payload.get("text", ""),
            "timestamp": ts_ms,
        }

    if frame_type == "transcript.final":
        # Committed user transcript turn
        return {
            "type": "transcript_commit",
            "speaker": "user",
            "text": payload.get("text", ""),
            "timestamp": ts_ms,
        }

    if frame_type == "transcript.ai_final":
        # Committed AI response turn
        return {
            "type": "transcript_commit",
            "speaker": "ai",
            "text": payload.get("text", ""),
            "timestamp": ts_ms,
        }

    if frame_type == "audio.response.end":
        return {"type": "audio_stream_end"}

    if frame_type == "barge_in.ack":
        return {"type": "barge_in_ack"}

    if frame_type == "error":
        return {
            "type": "error",
            "code": payload.get("code", "UNKNOWN_ERROR"),
            "message": payload.get("message", "An error occurred"),
        }

    if frame_type == "pipeline.state":
        # TOD pipeline visualization — forward to the frontend
        # Ensure the state shape matches FE PipelineStateSchema expectations
        raw_state = payload.get("state") or {}
        actual_session_id = frame.get("session_id", "")
        return {
            "type": "pipeline_state",
            "session_id": str(actual_session_id),
            "stt_text": payload.get("stt_text"),
            "nlu": payload.get("nlu"),
            "state": {
                "session_id": str(actual_session_id),
                "intent": raw_state.get("intent") or raw_state.get("current_intent"),
                "intent_confidence": raw_state.get("intent_confidence", 0.0),
                "slots": raw_state.get("slots", {}),
                "confirmed": raw_state.get("confirmed", False),
                "turn_count": raw_state.get("turn_count", 0),
            },
            "action": payload.get("action"),
            "target_slot": payload.get("target_slot"),
            "nlg_response": payload.get("nlg_response", ""),
        }

    if frame_type == "audio.response.start":
        # Transition FE to 'speaking' so barge-in detection and UI update work
        return {
            "type": "call_state",
            "state": "speaking",
            "timestamp": ts_ms,
        }

    if frame_type == "transcript.low_confidence":
        # Surface the low-confidence prompt as an AI transcript commit
        # so the user sees the system's "please repeat" message
        return {
            "type": "transcript_commit",
            "speaker": "ai",
            "text": payload.get("prompt_message", ""),
            "timestamp": ts_ms,
        }

    # Unrecognised frame types — drop silently
    return None


async def _send_adapted(websocket: WebSocket, frame: dict[str, Any]) -> None:
    """Adapt a BE frame and send it; silently drop frames with no FE equivalent."""
    adapted = _adapt_outbound_frame(frame)
    if adapted is not None:
        await _send_text(websocket, adapted)


async def _send_error(
    websocket: WebSocket,
    session_id: str,
    code: str,
    message: str,
    *,
    recoverable: bool = False,
) -> None:
    # FE expects flat { type, code, message } — no payload wrapper, no session_id
    await _send_text(
        websocket,
        {"type": "error", "code": code, "message": message},
    )


async def _send_session_state(
    websocket: WebSocket,
    session_id: str,
    state: str,
    previous_state: str | None = None,
) -> None:
    # Convert BE state to FE state then send as ``call_state``
    mapped_state = _STATE_MAP.get(state, state)
    import time as _t

    await _send_text(
        websocket,
        {
            "type": "call_state",
            "state": mapped_state,
            "timestamp": int(_t.time() * 1000),
        },
    )


# ── WebSocket route ───────────────────────────────────────────────────────────


@router.websocket("/ws/calls/{session_id}")
async def ws_call_handler(
    websocket: WebSocket,
    session_id: str,
    handle_call: HandleCallUseCase = Depends(get_handle_call),  # noqa: B008
) -> None:
    """WebSocket call controller — audio frame dispatch and response streaming."""
    # ── Auth: validate JWT on upgrade ──────────────────────────────────────
    token = _extract_token(websocket)
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    from fastapi import HTTPException

    from src.interface.rest.sessions import verify_jwt

    try:
        jwt_session_id = verify_jwt(token)
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # ── Validate session_id UUID ───────────────────────────────────────────
    try:
        parsed_session_id = uuid.UUID(session_id)
    except ValueError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if jwt_session_id != str(parsed_session_id):
        # Token was issued for a different session
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # ── Verify session exists in the call handler state ───────────────────
    # (The session was already created via POST /sessions; if it's not in
    #  the use case's queue map it means this is a fresh WS connect)
    handle_call.ensure_session_ready(session_id)

    # Clear stale dialogue state on reconnect so intent/slots don't carry over
    handle_call.clear_pipeline_state(session_id)

    # ── Accept the WebSocket connection ────────────────────────────────────
    await websocket.accept()
    structlog.contextvars.bind_contextvars(session_id=session_id)
    logger.info("ws_connected", session_id=session_id)

    # Mark presence (graceful on Redis failure)
    with contextlib.suppress(Exception):
        await handle_call.mark_present(session_id)

    # Notify client that we're listening
    await _send_session_state(websocket, session_id, "listening")

    # ── Frame dispatch loop ────────────────────────────────────────────────
    try:
        while True:
            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"] is not None:
                # Binary audio frame — raw PCM bytes
                await handle_call.handle_audio_chunk(session_id, data["bytes"])
                continue

            raw_text = data.get("text", "")
            if not raw_text:
                continue

            # Quick-parse FE simple control frames before full validation
            try:
                quick = json.loads(raw_text)
                fe_type = quick.get("type") if isinstance(quick, dict) else None
            except Exception:
                fe_type = None

            if fe_type == "call_start":
                # FE initiates the call — transition to listening (already sent on connect)
                logger.info("call_start_received", session_id=session_id)
                await _send_session_state(websocket, session_id, "listening")
                continue

            if fe_type == "call_end":
                # FE-initiated graceful teardown
                await _teardown(websocket, session_id, handle_call, graceful=True)
                break

            if fe_type == "audio_end":
                # VAD detected end-of-speech — trigger the STT→TOD→TTS pipeline
                logger.info("audio_end_vad", session_id=session_id)
                await _send_session_state(websocket, session_id, "ai_thinking", "listening")
                await handle_call.handle_audio_end(
                    session_id,
                    send_text=lambda d: _send_adapted(websocket, d),
                    send_binary=lambda b: _send_binary(websocket, b),
                )
                await _send_session_state(websocket, session_id, "listening", "ai_speaking")
                continue

            if fe_type == "barge_in":
                # FE detected voice activity during AI speech — publish Redis cancel signal
                logger.debug("barge_in_frame_received", session_id=session_id)
                await handle_call.publish_barge_in(session_id)
                await _send_text(websocket, {"type": "barge_in_ack"})
                continue

            # Parse and dispatch structured text frame (audio.chunk, audio.end, session.end, etc.)
            await _dispatch_text_frame(
                raw_text=raw_text,
                websocket=websocket,
                session_id=session_id,
                handle_call=handle_call,
            )

    except WebSocketDisconnect:
        logger.info("ws_disconnect", session_id=session_id)
    except Exception:
        logger.exception("ws_unexpected_error", session_id=session_id)
    finally:
        # Same teardown path as session.end
        await _teardown(websocket, session_id, handle_call)


async def _dispatch_text_frame(
    *,
    raw_text: str,
    websocket: WebSocket,
    session_id: str,
    handle_call: HandleCallUseCase,
) -> None:
    """Parse a text frame and route to the appropriate handler."""
    try:
        frame_dict = json.loads(raw_text)
    except json.JSONDecodeError:
        await _send_error(
            websocket, session_id, "INVALID_PAYLOAD", "Frame is not valid JSON"
        )
        return

    try:
        frame = InboundWSFrame.model_validate(frame_dict)  # type: ignore[call-arg]
    except ValidationError as exc:
        await _send_error(
            websocket,
            session_id,
            "INVALID_PAYLOAD",
            f"Frame validation failed: {exc.error_count()} error(s)",
        )
        return

    frame_type = frame.type  # type: ignore[union-attr]

    if frame_type == "audio.chunk":
        # Validate audio frame payload
        try:
            audio_bytes = validate_audio_chunk(frame.payload)  # type: ignore[union-attr]
        except PayloadValidationError as exc:
            await _send_error(websocket, session_id, "INVALID_PAYLOAD", str(exc))
            return
        await handle_call.handle_audio_chunk(session_id, audio_bytes)

    elif frame_type == "audio.end":
        await _send_session_state(websocket, session_id, "ai_thinking", "listening")
        # Wrap send_text with the adapter so all BE-internal frames are
        # translated to the FE-expected flat format before being sent over the wire.
        await handle_call.handle_audio_end(
            session_id,
            send_text=lambda d: _send_adapted(websocket, d),
            send_binary=lambda b: _send_binary(websocket, b),
        )
        await _send_session_state(websocket, session_id, "listening", "ai_speaking")

    elif frame_type == "session.resume":
        # Resume: validate session is still active and re-announce listening state
        await _send_session_state(websocket, session_id, "listening")
        logger.info(
            "session_resumed",
            session_id=session_id,
            last_sequence=frame.payload.last_sequence,  # type: ignore[union-attr]
        )

    elif frame_type == "session.end":
        await _teardown(websocket, session_id, handle_call, graceful=True)


async def _teardown(
    websocket: WebSocket,
    session_id: str,
    handle_call: HandleCallUseCase,
    *,
    graceful: bool = False,
) -> None:
    """Perform session teardown and optionally close the WebSocket."""
    try:
        await handle_call.teardown(session_id)
    except Exception:
        logger.exception("teardown_error", session_id=session_id)

    with contextlib.suppress(Exception):
        if graceful:
            await _send_session_state(websocket, session_id, "call_ended")
            await websocket.close(code=1000)

    logger.info("ws_teardown_complete", session_id=session_id)
