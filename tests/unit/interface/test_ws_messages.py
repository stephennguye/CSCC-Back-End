"""Tests for WebSocket message models (inbound and outbound frames)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.interface.dtos.ws_messages import (
    AudioChunkFrame,
    AudioChunkPayload,
    AudioEndFrame,
    AudioEndPayload,
    AudioResponseEndFrame,
    AudioResponseStartFrame,
    BargeInAckFrame,
    ErrorFrame,
    ErrorPayload,
    PipelineStateFrame,
    PipelineStateNLU,
    PipelineStateDialogue,
    PipelineStatePayload,
    SessionEndFrame,
    SessionResumeFrame,
    SessionStateFrame,
    SessionStatePayload,
    TranscriptFinalFrame,
    TranscriptPartialFrame,
)


_SID = uuid.uuid4()


class TestInboundAudioChunkFrame:
    def test_valid_pcm(self) -> None:
        frame = AudioChunkFrame(
            type="audio.chunk",
            session_id=_SID,
            payload=AudioChunkPayload(
                sequence=1, codec="pcm_16khz_mono", data="AAAA"
            ),
        )
        assert frame.type == "audio.chunk"
        assert frame.payload.sequence == 1
        assert frame.payload.codec == "pcm_16khz_mono"

    def test_valid_opus(self) -> None:
        frame = AudioChunkFrame(
            type="audio.chunk",
            session_id=_SID,
            payload=AudioChunkPayload(
                sequence=1, codec="opus_48khz", data="AAAA"
            ),
        )
        assert frame.payload.codec == "opus_48khz"

    def test_invalid_codec_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AudioChunkFrame(
                type="audio.chunk",
                session_id=_SID,
                payload=AudioChunkPayload(
                    sequence=1, codec="mp3", data="AAAA"
                ),
            )

    def test_zero_sequence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AudioChunkFrame(
                type="audio.chunk",
                session_id=_SID,
                payload=AudioChunkPayload(
                    sequence=0, codec="pcm_16khz_mono", data="AAAA"
                ),
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AudioChunkPayload(
                sequence=1, codec="pcm_16khz_mono", data="AAAA", extra="bad"
            )


class TestInboundAudioEndFrame:
    def test_valid(self) -> None:
        frame = AudioEndFrame(
            type="audio.end",
            session_id=_SID,
            payload=AudioEndPayload(sequence=5),
        )
        assert frame.type == "audio.end"
        assert frame.payload.sequence == 5


class TestInboundSessionResumeFrame:
    def test_valid(self) -> None:
        frame = SessionResumeFrame(
            type="session.resume",
            session_id=_SID,
            payload={"last_sequence": 3},
        )
        assert frame.payload.last_sequence == 3

    def test_negative_last_sequence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SessionResumeFrame(
                type="session.resume",
                session_id=_SID,
                payload={"last_sequence": -1},
            )


class TestInboundSessionEndFrame:
    def test_valid_empty_payload(self) -> None:
        frame = SessionEndFrame(type="session.end", session_id=_SID)
        assert frame.type == "session.end"


class TestOutboundTranscriptFrames:
    def test_partial(self) -> None:
        frame = TranscriptPartialFrame(
            session_id=_SID,
            payload={"text": "Tôi muốn", "confidence": 0.8, "segment_id": "seg-1"},
        )
        assert frame.type == "transcript.partial"
        assert frame.payload.text == "Tôi muốn"

    def test_final(self) -> None:
        frame = TranscriptFinalFrame(
            session_id=_SID,
            payload={"text": "Tôi muốn bay", "confidence": 0.95, "segment_id": "seg-1"},
        )
        assert frame.type == "transcript.final"

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptPartialFrame(
                session_id=_SID,
                payload={"text": "test", "confidence": 1.5, "segment_id": "seg-1"},
            )


class TestOutboundSessionStateFrame:
    def test_valid_states(self) -> None:
        for state_val in ["connecting", "listening", "ai_thinking", "ai_speaking", "call_ended", "error"]:
            frame = SessionStateFrame(
                session_id=_SID,
                payload=SessionStatePayload(
                    state=state_val,
                    timestamp=datetime.now(tz=UTC),
                ),
            )
            assert frame.payload.state == state_val


class TestOutboundErrorFrame:
    def test_valid(self) -> None:
        frame = ErrorFrame(
            session_id=_SID,
            payload=ErrorPayload(
                code="LLM_TIMEOUT",
                message="Request timed out",
                recoverable=True,
            ),
        )
        assert frame.type == "error"
        assert frame.payload.recoverable is True


class TestOutboundBargeInAckFrame:
    def test_valid(self) -> None:
        frame = BargeInAckFrame(
            session_id=_SID,
            payload={"halted_turn_id": "t-1"},
        )
        assert frame.type == "barge_in.ack"


class TestOutboundAudioResponseFrames:
    def test_start(self) -> None:
        frame = AudioResponseStartFrame(
            session_id=_SID,
            payload={"turn_id": "t-1"},
        )
        assert frame.type == "audio.response.start"
        assert frame.payload.codec == "pcm_16khz_mono"  # default

    def test_end(self) -> None:
        frame = AudioResponseEndFrame(
            session_id=_SID,
            payload={"turn_id": "t-1"},
        )
        assert frame.type == "audio.response.end"


class TestOutboundPipelineStateFrame:
    def test_valid(self) -> None:
        frame = PipelineStateFrame(
            session_id=_SID,
            payload=PipelineStatePayload(
                turn_id="t-1",
                nlu=PipelineStateNLU(
                    intent="atis_flight", confidence=0.9, slots={"fromloc.city_name": "Hà Nội"}
                ),
                state=PipelineStateDialogue(
                    session_id="s1",
                    intent="atis_flight",
                    intent_confidence=0.9,
                    slots={"fromloc.city_name": "Hà Nội"},
                    confirmed=False,
                    turn_count=1,
                ),
                action="request_slot",
                target_slot="toloc.city_name",
                timestamp="2026-03-18T12:00:00Z",
            ),
        )
        assert frame.type == "pipeline.state"
        assert frame.payload.nlu.intent == "atis_flight"
