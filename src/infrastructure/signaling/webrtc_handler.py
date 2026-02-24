"""WebRTC SFU signaling stub.

No-op async methods serving as a placeholder for future Mediasoup / LiveKit
integration.  All methods are safe to call and will return default values
without side effects.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class WebRTCSignalingHandler:
    """Placeholder WebRTC SFU signaling handler.

    Future implementation will delegate to Mediasoup or LiveKit to manage
    server-side media routing for WebRTC peers.  Until then every method
    is a no-op that logs a debug trace so callers can see the integration
    points without any failures.
    """

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    async def create_transport(self, session_id: str) -> dict:
        """Allocate a new WebRTC transport for the given session.

        Returns a stub transport descriptor dict.
        """
        logger.debug("webrtc_create_transport_stub", session_id=session_id)
        return {"transport_id": None, "stub": True}

    async def close_transport(self, session_id: str) -> None:
        """Release all WebRTC resources associated with *session_id*."""
        logger.debug("webrtc_close_transport_stub", session_id=session_id)

    # ------------------------------------------------------------------ #
    # ICE / DTLS negotiation                                               #
    # ------------------------------------------------------------------ #

    async def handle_offer(self, session_id: str, sdp_offer: str) -> str:
        """Process an SDP offer and return an SDP answer.

        Currently returns an empty string (no real signaling).
        """
        logger.debug("webrtc_handle_offer_stub", session_id=session_id)
        return ""

    async def add_ice_candidate(
        self,
        session_id: str,
        candidate: dict,
    ) -> None:
        """Register an ICE candidate for *session_id*."""
        logger.debug("webrtc_add_ice_candidate_stub", session_id=session_id)

    # ------------------------------------------------------------------ #
    # Media track management                                               #
    # ------------------------------------------------------------------ #

    async def produce_audio(self, session_id: str, rtp_parameters: dict) -> str | None:
        """Register an audio producer (caller → server).  Returns producer_id."""
        logger.debug("webrtc_produce_audio_stub", session_id=session_id)
        return None

    async def consume_audio(self, session_id: str, producer_id: str) -> dict:
        """Create an audio consumer (server → caller).  Returns consumer descriptor."""
        logger.debug(
            "webrtc_consume_audio_stub",
            session_id=session_id,
            producer_id=producer_id,
        )
        return {"consumer_id": None, "stub": True}

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        """Return True — stub is always healthy."""
        return True
