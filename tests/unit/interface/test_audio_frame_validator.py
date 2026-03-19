"""Tests for src.interface.validators.audio_frame."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import pytest

from src.domain.errors import PayloadValidationError
from src.interface.validators.audio_frame import (
    ALLOWED_CODECS,
    MAX_DECODED_BYTES,
    validate_audio_chunk,
)


@dataclass
class FakeAudioChunkPayload:
    """Mimics AudioChunkPayload for testing without Pydantic dependency."""

    sequence: int
    codec: str
    data: str


def _make_payload(
    *,
    sequence: int = 1,
    codec: str = "pcm_16khz_mono",
    raw_bytes: bytes | None = None,
) -> FakeAudioChunkPayload:
    if raw_bytes is None:
        raw_bytes = b"\x00" * 100
    return FakeAudioChunkPayload(
        sequence=sequence,
        codec=codec,
        data=base64.b64encode(raw_bytes).decode(),
    )


class TestValidPayloads:
    def test_valid_pcm(self) -> None:
        payload = _make_payload(codec="pcm_16khz_mono")
        result = validate_audio_chunk(payload)
        assert isinstance(result, bytes)
        assert len(result) == 100

    def test_opus_rejected(self) -> None:
        """opus_48khz is no longer accepted — STT only supports PCM."""
        payload = _make_payload(codec="opus_48khz")
        with pytest.raises(PayloadValidationError, match="Unsupported audio codec"):
            validate_audio_chunk(payload)

    def test_exact_max_size(self) -> None:
        payload = _make_payload(raw_bytes=b"\x00" * MAX_DECODED_BYTES)
        result = validate_audio_chunk(payload)
        assert len(result) == MAX_DECODED_BYTES


class TestCodecValidation:
    def test_invalid_codec_raises(self) -> None:
        payload = _make_payload(codec="mp3")
        with pytest.raises(PayloadValidationError, match="Unsupported audio codec"):
            validate_audio_chunk(payload)

    def test_empty_codec_raises(self) -> None:
        payload = _make_payload(codec="")
        with pytest.raises(PayloadValidationError):
            validate_audio_chunk(payload)


class TestSequenceValidation:
    def test_zero_sequence_raises(self) -> None:
        payload = _make_payload(sequence=0)
        with pytest.raises(PayloadValidationError, match="must be > 0"):
            validate_audio_chunk(payload)

    def test_negative_sequence_raises(self) -> None:
        payload = _make_payload(sequence=-1)
        with pytest.raises(PayloadValidationError):
            validate_audio_chunk(payload)


class TestBase64Validation:
    def test_invalid_base64_raises(self) -> None:
        payload = FakeAudioChunkPayload(sequence=1, codec="pcm_16khz_mono", data="not!valid!base64")
        with pytest.raises(PayloadValidationError, match="not valid base64"):
            validate_audio_chunk(payload)


class TestSizeValidation:
    def test_oversized_payload_raises(self) -> None:
        payload = _make_payload(raw_bytes=b"\x00" * (MAX_DECODED_BYTES + 1))
        with pytest.raises(PayloadValidationError, match="exceeds"):
            validate_audio_chunk(payload)
