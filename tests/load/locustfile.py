"""Locust load test for AI Call Center Backend.

Scenarios
---------
- **WebSocketCaller**: Simulates a caller opening a WebSocket call session.
  1. POST /api/v1/sessions   -> obtain session_id + JWT token
  2. Connect ws://host/ws/calls/{session_id} with Bearer token
  3. Send 3x audio.chunk frames (synthetic PCM-like base64 payload)
  4. Send audio.end frame
  5. Wait for audio.response.end (or timeout 10 s = 1.5 s budget + headroom)
  6. Send session.end → disconnect
  7. Record first-token latency (response.token received) as a custom metric

- **RESTHistoryReader**: Simulates support staff polling conversation history
  after calls complete.

Performance targets (from spec, SC-001 / SC-002)
-------------------------------------------------
- First response token received within 1.5 s (p95) — asserted in custom event
- No latency degradation under 100 concurrent callers

Usage
-----
  locust -f tests/load/locustfile.py \
         --host http://localhost:8000 \
         --users 100 \
         --spawn-rate 10 \
         --run-time 5m \
         --headless --only-summary

Environment variables
---------------------
LOCUST_WS_TIMEOUT   — seconds to wait for audio.response.end (default: 10)
LOCUST_TOKEN_BUDGET — first-token latency budget in seconds (default: 1.5)
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from typing import Any

from locust import FastHttpUser, between, events, task  # type: ignore[import-untyped]

try:
    import websocket  # type: ignore[import-untyped]

    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

_WS_TIMEOUT: float = float(os.environ.get("LOCUST_WS_TIMEOUT", "10.0"))
_TOKEN_BUDGET: float = float(os.environ.get("LOCUST_TOKEN_BUDGET", "1.5"))

# Synthetic 4 KB chunk of silence-like PCM16 data (all zeros)
_SILENT_PCM_B64: str = base64.b64encode(b"\x00" * 4000).decode()


def _make_audio_chunk(session_id: str, sequence: int) -> str:
    return json.dumps(
        {
            "type": "audio.chunk",
            "session_id": session_id,
            "payload": {
                "sequence": sequence,
                "codec": "pcm_16khz_mono",
                "data": _SILENT_PCM_B64,
            },
        }
    )


def _make_audio_end(session_id: str, sequence: int) -> str:
    return json.dumps(
        {
            "type": "audio.end",
            "session_id": session_id,
            "payload": {"sequence": sequence},
        }
    )


def _make_session_end(session_id: str) -> str:
    return json.dumps(
        {
            "type": "session.end",
            "session_id": session_id,
            "payload": {},
        }
    )


class WebSocketCaller(FastHttpUser):
    """Simulates a caller making a full WebSocket call.

    100 concurrent users with a short wait between calls mirrors the SC-002
    concurrency requirement.
    """

    wait_time = between(1, 3)

    @task(weight=5)
    def complete_call(self) -> None:
        """Full call flow: create session → WS call → teardown."""
        # ── Step 1: Create session ────────────────────────────────────────
        with self.client.post(
            "/api/v1/sessions",
            json={},
            name="POST /api/v1/sessions",
            catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 201):
                resp.failure(f"Session creation failed: {resp.status_code}")
                return
            data: dict[str, Any] = resp.json()
            session_id: str = data.get("session_id", "")
            token: str = data.get("token", "")

        if not session_id or not token:
            events.request.fire(
                request_type="WS",
                name="ws_call",
                response_time=0,
                response_length=0,
                exception=ValueError("No session_id or token in response"),
                context={},
            )
            return

        if not _WS_AVAILABLE:
            # Graceful skip when websocket-client not installed
            events.request.fire(
                request_type="WS",
                name="ws_call",
                response_time=0,
                response_length=0,
                exception=ImportError("websocket-client not installed"),
                context={},
            )
            return

        # ── Step 2: WebSocket call ─────────────────────────────────────────
        host = self.host.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{host}/ws/calls/{session_id}"

        start_time = time.perf_counter()
        first_token_time: float | None = None
        audio_end_received = False
        error: Exception | None = None

        try:
            ws = websocket.create_connection(
                ws_url,
                header={"Authorization": f"Bearer {token}"},
                timeout=_WS_TIMEOUT,
            )

            # Send 3 audio chunks
            for seq in range(1, 4):
                ws.send(_make_audio_chunk(session_id, seq))

            # Signal end of audio
            ws.send(_make_audio_end(session_id, 3))

            # Receive frames until audio.response.end or timeout
            deadline = time.perf_counter() + _WS_TIMEOUT
            while time.perf_counter() < deadline:
                ws.settimeout(max(0.1, deadline - time.perf_counter()))
                try:
                    raw = ws.recv()
                except Exception:
                    break
                if not isinstance(raw, str):
                    continue
                try:
                    frame: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                frame_type = frame.get("type", "")

                if frame_type == "response.token" and first_token_time is None:
                    first_token_time = time.perf_counter() - start_time

                if frame_type == "audio.response.end":
                    audio_end_received = True
                    break

                if frame_type == "error":
                    error = RuntimeError(f"Server error frame: {frame}")
                    break

            # Teardown
            ws.send(_make_session_end(session_id))
            ws.close()

        except Exception as exc:
            error = exc

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # ── Report first-token latency metric ──────────────────────────────
        if first_token_time is not None:
            events.request.fire(
                request_type="WS",
                name="ws_first_token_latency",
                response_time=int(first_token_time * 1000),
                response_length=0,
                exception=None,
                context={},
            )
            # Assert budget
            if first_token_time > _TOKEN_BUDGET:
                events.request.fire(
                    request_type="WS",
                    name="ws_first_token_budget_exceeded",
                    response_time=int(first_token_time * 1000),
                    response_length=0,
                    exception=RuntimeError(
                        f"First-token latency {first_token_time:.3f}s "
                        f"exceeds {_TOKEN_BUDGET}s budget (SC-001)"
                    ),
                    context={},
                )

        # ── Report overall call metric ─────────────────────────────────────
        events.request.fire(
            request_type="WS",
            name="ws_call",
            response_time=elapsed_ms,
            response_length=0,
            exception=error if not audio_end_received else None,
            context={},
        )


class RESTHistoryReader(FastHttpUser):
    """Simulates support staff polling history / health after calls."""

    wait_time = between(2, 5)

    @task(weight=2)
    def health_check(self) -> None:
        self.client.get("/api/v1/health", name="GET /api/v1/health")

    @task(weight=1)
    def get_history_not_found(self) -> None:
        """Poll history for a non-existent session (expected 404)."""
        random_id = str(uuid.uuid4())
        with self.client.get(
            f"/api/v1/conversations/{random_id}/history",
            name="GET /api/v1/conversations/{id}/history",
            catch_response=True,
        ) as resp:
            if resp.status_code == 404:
                resp.success()
            elif resp.status_code != 200:
                resp.failure(f"Unexpected status: {resp.status_code}")
