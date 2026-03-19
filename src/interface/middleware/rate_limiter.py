"""Sliding-window rate limiting middleware (FR-024).

Enforces per-IP request rate limits on all REST endpoints except ``/health``.
Uses Redis counters incremented via :class:`~src.infrastructure.cache.redis_client.RedisClient`.

Configuration (environment variables):
    - ``RATE_LIMIT_SESSIONS``       — requests / window for ``/api/v1/sessions``       (default 20)
    - ``RATE_LIMIT_CONVERSATIONS``  — requests / window for ``/api/v1/conversations/*`` (default 60)
    - ``RATE_LIMIT_DIALOGUE``       — requests / window for ``/api/v1/dialogue/*``      (default 30)
    - ``RATE_LIMIT_WINDOW_SECONDS`` — sliding-window duration in seconds                (default 60)

When a rate limit is exceeded the middleware returns a ``429 Too Many Requests``
response with the following headers:
    - ``Retry-After``           — seconds until the current window resets
    - ``X-RateLimit-Limit``     — total requests allowed per window
    - ``X-RateLimit-Remaining`` — remaining requests in the current window
    - ``X-RateLimit-Reset``     — UTC epoch seconds when the window resets
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request

logger = structlog.get_logger(__name__)

# ── Limit configuration ───────────────────────────────────────────────────────

_LIMIT_SESSIONS: int = int(os.environ.get("RATE_LIMIT_SESSIONS", "20"))
_LIMIT_CONVERSATIONS: int = int(os.environ.get("RATE_LIMIT_CONVERSATIONS", "60"))
_LIMIT_DIALOGUE: int = int(os.environ.get("RATE_LIMIT_DIALOGUE", "30"))

# Endpoints exempt from rate limiting
_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/api/v1/health",
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _get_limit_for_path(path: str) -> tuple[int, str] | None:
    """Return ``(limit, group_key)`` for *path*, or ``None`` if exempt.

    The group key is used as the Redis key suffix so that all paths in the
    same group share a single counter per IP.
    """
    if path in _EXEMPT_PATHS or path.startswith("/ws/"):
        return None
    if path.startswith("/api/v1/sessions"):
        return _LIMIT_SESSIONS, "sessions"
    if path.startswith("/api/v1/conversations"):
        return _LIMIT_CONVERSATIONS, "conversations"
    if path.startswith("/api/v1/dialogue"):
        return _LIMIT_DIALOGUE, "dialogue"
    # Default for any other /api/v1/* path: use conversations limit
    if path.startswith("/api/v1/"):
        return _LIMIT_CONVERSATIONS, "api_default"
    return None


def _client_ip(request: Request) -> str:
    """Extract the real client IP, honouring X-Forwarded-For when present."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the leftmost (original client) address
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding-window rate limiting backed by Redis.

    Attach to a FastAPI/Starlette app via::

        from starlette.middleware.base import BaseHTTPMiddleware
        app.add_middleware(RateLimitMiddleware)

    The middleware reads the Redis client from ``app.state.redis`` at
    request time, so it must be registered *after* the lifespan startup
    has wired ``app.state.redis``.  If Redis is unavailable the middleware
    degrades gracefully — requests are allowed through rather than blocked.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        limit_pair = _get_limit_for_path(path)

        if limit_pair is None:
            # Exempt path — pass through immediately
            return await call_next(request)

        limit, group = limit_pair
        ip = _client_ip(request)
        redis: Any = getattr(request.app.state, "redis", None)

        if redis is None:
            # Infrastructure not ready (e.g. startup in progress).
            # Degrade gracefully — let the request through.
            logger.debug("rate_limit_redis_unavailable", path=path, ip=ip)
            return await call_next(request)

        try:
            count: int = await redis.increment_rate_limit(ip, group)
            ttl: int = await redis.get_rate_limit_ttl(ip, group)
        except Exception:
            # Redis error — degrade gracefully.
            logger.warning("rate_limit_redis_error", path=path, ip=ip)
            return await call_next(request)

        remaining = max(limit - count, 0)
        reset_epoch = int(time.time()) + max(ttl, 0)

        if count > limit:
            retry_after = max(ttl, 1)
            logger.info(
                "rate_limit_exceeded",
                ip=ip,
                group=group,
                count=count,
                limit=limit,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": (
                            f"Too many requests. Retry after {retry_after} seconds."
                        ),
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_epoch),
                    "Content-Type": "application/json",
                },
            )

        # Within limit — continue and inject rate-limit headers into response.
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_epoch)
        return response
