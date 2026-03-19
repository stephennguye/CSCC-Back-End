"""Contract tests for WebSocket message schemas.

Validates that all inbound and outbound WebSocket frame types
conform to their Pydantic v2 schemas (strict mode, no extra fields).

Uses model_validate_json() to simulate real wire format (JSON strings),
which correctly handles UUID coercion from string that strict mode blocks
with model_validate(dict).
"""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import ValidationError

from src.interface.dtos.ws_messages import (
    AudioChunkFrame,
    AudioEndFrame,
    AudioResponseEndFrame,
    AudioResponseFrame,
    AudioResponseStartFrame,
    BargeInAckFrame,
    ErrorFrame,
    PipelineStateFrame,
    SessionEndFrame,
    SessionResumeFrame,
    SessionStateFrame,
    TranscriptAIFinalFrame,
    TranscriptFinalFrame,
    TranscriptLowConfidenceFrame,
    TranscriptPartialFrame,
)


_SID = str(uuid.uuid4())


def _j(d: dict) -> str:
    return json.dumps(d)


# ══════════════════════════════════════════════════════════════════════════════
# Inbound frame contracts (Client → Server)
# ══════════════════════════════════════════════════════════════════════════════


class TestInboundAudioChunkContract:
    def test_valid_pcm_contract(self) -> None:
        frame = AudioChunkFrame.model_validate_json(_j({
            "type": "audio.chunk", "session_id": _SID,
            "payload": {"sequence": 1, "codec": "pcm_16khz_mono", "data": "AAAA"},
        }))
        assert frame.type == "audio.chunk"
        assert frame.payload.sequence == 1

    def test_valid_opus_contract(self) -> None:
        frame = AudioChunkFrame.model_validate_json(_j({
            "type": "audio.chunk", "session_id": _SID,
            "payload": {"sequence": 1, "codec": "opus_48khz", "data": "AAAA"},
        }))
        assert frame.payload.codec == "opus_48khz"

    def test_rejects_unknown_codec(self) -> None:
        with pytest.raises(ValidationError):
            AudioChunkFrame.model_validate_json(_j({
                "type": "audio.chunk", "session_id": _SID,
                "payload": {"sequence": 1, "codec": "wav", "data": "AAAA"},
            }))

    def test_rejects_zero_sequence(self) -> None:
        with pytest.raises(ValidationError):
            AudioChunkFrame.model_validate_json(_j({
                "type": "audio.chunk", "session_id": _SID,
                "payload": {"sequence": 0, "codec": "pcm_16khz_mono", "data": "AAAA"},
            }))

    def test_rejects_extra_fields_in_payload(self) -> None:
        with pytest.raises(ValidationError):
            AudioChunkFrame.model_validate_json(_j({
                "type": "audio.chunk", "session_id": _SID,
                "payload": {"sequence": 1, "codec": "pcm_16khz_mono", "data": "AAAA", "extra": "bad"},
            }))

    def test_rejects_missing_data(self) -> None:
        with pytest.raises(ValidationError):
            AudioChunkFrame.model_validate_json(_j({
                "type": "audio.chunk", "session_id": _SID,
                "payload": {"sequence": 1, "codec": "pcm_16khz_mono"},
            }))


class TestInboundAudioEndContract:
    def test_valid(self) -> None:
        frame = AudioEndFrame.model_validate_json(_j({
            "type": "audio.end", "session_id": _SID,
            "payload": {"sequence": 10},
        }))
        assert frame.payload.sequence == 10

    def test_rejects_negative_sequence(self) -> None:
        with pytest.raises(ValidationError):
            AudioEndFrame.model_validate_json(_j({
                "type": "audio.end", "session_id": _SID,
                "payload": {"sequence": -1},
            }))


class TestInboundSessionResumeContract:
    def test_valid(self) -> None:
        frame = SessionResumeFrame.model_validate_json(_j({
            "type": "session.resume", "session_id": _SID,
            "payload": {"last_sequence": 5},
        }))
        assert frame.payload.last_sequence == 5

    def test_allows_zero_last_sequence(self) -> None:
        frame = SessionResumeFrame.model_validate_json(_j({
            "type": "session.resume", "session_id": _SID,
            "payload": {"last_sequence": 0},
        }))
        assert frame.payload.last_sequence == 0


class TestInboundSessionEndContract:
    def test_valid_empty_payload(self) -> None:
        frame = SessionEndFrame.model_validate_json(_j({
            "type": "session.end", "session_id": _SID,
        }))
        assert frame.type == "session.end"

    def test_valid_explicit_empty_payload(self) -> None:
        frame = SessionEndFrame.model_validate_json(_j({
            "type": "session.end", "session_id": _SID, "payload": {},
        }))
        assert frame.type == "session.end"


# ══════════════════════════════════════════════════════════════════════════════
# Outbound frame contracts (Server → Client)
# ══════════════════════════════════════════════════════════════════════════════


class TestOutboundTranscriptContract:
    def test_partial(self) -> None:
        frame = TranscriptPartialFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"text": "Tôi muốn", "confidence": 0.8, "segment_id": "seg-1"},
        }))
        assert frame.type == "transcript.partial"

    def test_final(self) -> None:
        frame = TranscriptFinalFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"text": "Tôi muốn bay", "confidence": 0.95, "segment_id": "seg-1"},
        }))
        assert frame.type == "transcript.final"

    def test_low_confidence(self) -> None:
        frame = TranscriptLowConfidenceFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"segment_id": "seg-1", "prompt_message": "Xin lỗi, bạn nói lại?"},
        }))
        assert frame.type == "transcript.low_confidence"

    def test_ai_final(self) -> None:
        frame = TranscriptAIFinalFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"turn_id": "t-1", "text": "Xin chào!", "timestamp": "2026-03-18T12:00:00Z", "sequence_number": 1},
        }))
        assert frame.type == "transcript.ai_final"

    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptPartialFrame.model_validate_json(_j({
                "session_id": _SID,
                "payload": {"text": "test", "confidence": 1.5, "segment_id": "seg-1"},
            }))

    def test_rejects_confidence_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptFinalFrame.model_validate_json(_j({
                "session_id": _SID,
                "payload": {"text": "test", "confidence": -0.1, "segment_id": "seg-1"},
            }))


class TestOutboundSessionStateContract:
    @pytest.mark.parametrize("state_val", [
        "connecting", "listening", "ai_thinking",
        "ai_speaking", "call_ended", "reconnecting", "error",
    ])
    def test_all_valid_states(self, state_val: str) -> None:
        frame = SessionStateFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"state": state_val, "timestamp": "2026-03-18T12:00:00Z"},
        }))
        assert frame.payload.state == state_val

    def test_with_previous_state(self) -> None:
        frame = SessionStateFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"state": "ai_thinking", "previous_state": "listening", "timestamp": "2026-03-18T12:00:00Z"},
        }))
        assert frame.payload.previous_state == "listening"


class TestOutboundErrorContract:
    def test_valid_error(self) -> None:
        frame = ErrorFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"code": "LLM_TIMEOUT", "message": "Timed out", "recoverable": True},
        }))
        assert frame.payload.code == "LLM_TIMEOUT"
        assert frame.payload.recoverable is True

    def test_non_recoverable_error(self) -> None:
        frame = ErrorFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"code": "SESSION_ENDED", "message": "Ended", "recoverable": False},
        }))
        assert frame.payload.recoverable is False


class TestOutboundPipelineStateContract:
    def test_full_pipeline_state(self) -> None:
        frame = PipelineStateFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {
                "turn_id": "t-1",
                "nlu": {"intent": "atis_flight", "confidence": 0.92, "slots": {"fromloc.city_name": "Hà Nội"}},
                "state": {"session_id": "s1", "intent": "atis_flight", "intent_confidence": 0.92, "slots": {}, "confirmed": False, "turn_count": 1},
                "action": "confirm", "target_slot": None, "timestamp": "2026-03-18T12:00:00Z",
            },
        }))
        assert frame.payload.nlu.intent == "atis_flight"
        assert frame.payload.action == "confirm"


class TestOutboundAudioResponseContract:
    def test_start(self) -> None:
        frame = AudioResponseStartFrame.model_validate_json(_j({
            "session_id": _SID, "payload": {"turn_id": "t-1"},
        }))
        assert frame.payload.codec == "pcm_16khz_mono"

    def test_end(self) -> None:
        frame = AudioResponseEndFrame.model_validate_json(_j({
            "session_id": _SID, "payload": {"turn_id": "t-1"},
        }))
        assert frame.type == "audio.response.end"

    def test_audio_data_frame(self) -> None:
        frame = AudioResponseFrame.model_validate_json(_j({
            "session_id": _SID,
            "payload": {"sequence": 1, "turn_id": "t-1", "codec": "pcm_16khz_mono", "data": "AAAA"},
        }))
        assert frame.type == "audio.response"


class TestOutboundMiscContract:
    def test_barge_in_ack(self) -> None:
        frame = BargeInAckFrame.model_validate_json(_j({
            "session_id": _SID, "payload": {"halted_turn_id": "t-1"},
        }))
        assert frame.type == "barge_in.ack"
