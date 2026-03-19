"""GET /api/v1/health — operational status of all dependent services.

Response behaviour (per contracts/rest-api.md):
  - All services healthy          → 200  ``{"status": "healthy", ...}``
  - Non-critical service down     → 207  ``{"status": "degraded", ...}``
  - Critical service down         → 503  ``{"status": "degraded", ...}``

Critical services: PostgreSQL, Redis.
Non-critical: faster-whisper, TTS.

Notes:
  - Each probe is wrapped in asyncio.wait_for(timeout=4) so the endpoint
    always responds within ~5 seconds even when a dependency hangs.
  - This endpoint is EXEMPT from rate limiting (per contracts/rest-api.md).
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.interface.dtos.rest_responses import HealthResponse, ServiceStatus

logger = structlog.get_logger(__name__)

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(tags=["health"])

# Per-probe timeout in seconds
_PROBE_TIMEOUT: float = float(os.environ.get("HEALTH_PROBE_TIMEOUT_SECONDS", "4.0"))


# ────────────────────────────────────────────────────────────────────────────
# Probe helpers
# ────────────────────────────────────────────────────────────────────────────


async def _probe(coro: Any) -> tuple[bool, int, str | None]:  # noqa: ANN401
    """Run *coro* with a timeout; return (ok, latency_ms, error_message)."""
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(coro, timeout=_PROBE_TIMEOUT)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return True, latency_ms, None
    except TimeoutError:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return False, latency_ms, f"probe timed out after {_PROBE_TIMEOUT}s"
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return False, latency_ms, str(exc)


async def _probe_postgres(session_factory: Any) -> None:  # noqa: ANN401
    from sqlalchemy import text

    async with session_factory() as db_session:
        await db_session.execute(text("SELECT 1"))


async def _probe_redis(redis: Any) -> None:  # noqa: ANN401
    ok = await redis.ping()
    if not ok:
        msg = "Redis PING returned False"
        raise ConnectionError(msg)


async def _probe_openai() -> None:
    """Check OpenAI reachability: verify API key is configured."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        msg = "OPENAI_API_KEY not configured"
        raise OSError(msg)


async def _probe_huggingface() -> None:
    """Check HuggingFace reachability: verify API token is configured."""
    token = os.environ.get("HUGGINGFACE_API_TOKEN", "")
    if not token:
        msg = "HUGGINGFACE_API_TOKEN not configured"
        raise OSError(msg)


async def _probe_faster_whisper() -> None:
    """Check that faster-whisper is importable (local model, no network)."""

    def _check() -> None:
        import faster_whisper  # noqa: F401

    await asyncio.get_event_loop().run_in_executor(None, _check)


async def _probe_tts() -> None:
    """Check that at least one TTS backend (Coqui or edge-tts) is importable."""

    def _check() -> None:
        try:
            import TTS  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            import edge_tts  # type: ignore[import-untyped]  # noqa: F401

    await asyncio.get_event_loop().run_in_executor(None, _check)


# ── Endpoint ──────────────────────────────────────────────────────────────────


async def _instant_fail(reason: str) -> tuple[bool, int, str | None]:
    """Return an immediate failure result for uninitialised services."""
    return False, 0, reason


@router.get(
    "/health",
    summary="Operational status of all dependent services",
    responses={
        200: {"description": "All services healthy"},
        207: {"description": "Degraded — non-critical service down"},
        503: {"description": "Critical service unavailable"},
    },
)
async def get_health(request: Request) -> JSONResponse:
    """Return a health report for all infrastructure dependencies.

    Probes run concurrently and finish within ``HEALTH_PROBE_TIMEOUT_SECONDS``
    (default 4 s) so the endpoint always responds within ~5 seconds.
    """
    session_factory: Any = getattr(request.app.state, "session_factory", None)
    redis: Any = getattr(request.app.state, "redis", None)

    # Build probe coroutines — fall back to instant failure when the
    # infrastructure singleton was never initialised (e.g. during tests).
    pg_coro = (
        _probe(_probe_postgres(session_factory))
        if session_factory is not None
        else _instant_fail("session_factory not initialised")
    )
    redis_coro = (
        _probe(_probe_redis(redis))
        if redis is not None
        else _instant_fail("redis not initialised")
    )

    # ── Run all probes concurrently ───────────────────────────────────────────
    gathered: list[tuple[bool, int, str | None]] = list(
        await asyncio.gather(
            pg_coro,
            redis_coro,
            _probe(_probe_faster_whisper()),
            _probe(_probe_tts()),
        )
    )

    pg_ok, pg_ms, pg_err = gathered[0]
    redis_ok, redis_ms, redis_err = gathered[1]
    whisper_ok, whisper_ms, whisper_err = gathered[2]
    tts_ok, tts_ms, tts_err = gathered[3]

    # ── Determine overall status ──────────────────────────────────────────────
    # Critical: postgres, redis
    critical_down = (not pg_ok) or (not redis_ok)
    any_down = not all([pg_ok, redis_ok, whisper_ok, tts_ok])

    if critical_down:
        http_status = 503
        overall = "degraded"
    elif any_down:
        http_status = 207
        overall = "degraded"
    else:
        http_status = 200
        overall = "healthy"

    def _svc(ok: bool, latency_ms: int, error: str | None) -> ServiceStatus:
        return ServiceStatus(
            status="healthy" if ok else "unhealthy",
            latency_ms=latency_ms,
            error=error,
        )

    response_body = HealthResponse(
        status=overall,  # type: ignore[arg-type]
        timestamp=datetime.now(tz=UTC),
        services={
            "postgres": _svc(pg_ok, pg_ms, pg_err),
            "redis": _svc(redis_ok, redis_ms, redis_err),
            "faster_whisper": _svc(whisper_ok, whisper_ms, whisper_err),
            "tts": _svc(tts_ok, tts_ms, tts_err),
        },
    )

    logger.debug(
        "health_check_complete",
        status=overall,
        postgres=pg_ok,
        redis=redis_ok,
        faster_whisper=whisper_ok,
        tts=tts_ok,
    )

    return JSONResponse(
        status_code=http_status,
        content=response_body.model_dump(mode="json"),
    )
