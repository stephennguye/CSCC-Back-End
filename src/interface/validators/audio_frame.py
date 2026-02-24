"""Audio frame validator for inbound WebSocket audio frames.

Enforces:
- Codec whitelist: ``pcm_16khz_mono`` and ``opus_48khz`` only
- Base64-decoded payload size ≤ 4 KB
- Sequence number is a positive integer > 0
- Raises :class:`~src.domain.errors.PayloadValidationError` on any violation
"""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING

from src.domain.errors import PayloadValidationError

if TYPE_CHECKING:
    from src.interface.dtos.ws_messages import AudioChunkPayload

# ── Constants ──────────────────────────────────────────────────────────────────

ALLOWED_CODECS: frozenset[str] = frozenset({"pcm_16khz_mono", "opus_48khz"})
MAX_DECODED_BYTES: int = 4096  # 4 KB


# ── Validator ──────────────────────────────────────────────────────────────────


def validate_audio_chunk(payload: AudioChunkPayload) -> bytes:
    """Validate an inbound ``audio.chunk`` payload.

    Args:
        payload: The :class:`AudioChunkPayload` extracted from the WS frame.

    Returns:
        Decoded raw audio bytes ready to be forwarded to the STT pipeline.

    Raises:
        :class:`~src.domain.errors.PayloadValidationError`: on any constraint
            violation (unknown codec, oversized chunk, invalid sequence or
            invalid base64 encoding).
    """
    # 1. Codec whitelist
    if payload.codec not in ALLOWED_CODECS:
        raise PayloadValidationError(
            f"Unsupported audio codec '{payload.codec}'. "
            f"Allowed values: {sorted(ALLOWED_CODECS)}"
        )

    # 2. Sequence number must be a positive integer > 0 (already enforced by
    #    Pydantic ``Field(gt=0)`` but we double-check here for defence-in-depth)
    if payload.sequence <= 0:
        raise PayloadValidationError(
            f"audio.chunk.sequence must be > 0, got {payload.sequence}"
        )

    # 3. Decode base64 and size guard
    try:
        audio_bytes = base64.b64decode(payload.data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PayloadValidationError(
            f"audio.chunk.data is not valid base64: {exc}"
        ) from exc

    if len(audio_bytes) > MAX_DECODED_BYTES:
        raise PayloadValidationError(
            f"audio.chunk.data decoded size ({len(audio_bytes)} bytes) exceeds "
            f"the maximum allowed size of {MAX_DECODED_BYTES} bytes."
        )

    return audio_bytes
