"""aioredis client — manages all ephemeral session state in Redis.

Key schema
----------
``session:{id}:state``            Hash   turn_state, barge_in_channel
``session:{id}:presence``         String presence flag (value "1")
``session:{id}:buffer``           List   capped short-term conversation turns
``session:{id}:barge_in``         Pub/Sub channel name (not a key)
``rate_limit:{ip}:{endpoint}``    String sliding-window counter

All session keys share the same TTL (``SESSION_TTL_SECONDS``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

if TYPE_CHECKING:
    from redis.asyncio.client import PubSub

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

_REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# How long session keys live after the last activity (seconds)
SESSION_TTL_SECONDS: int = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))

# Maximum number of turns kept in the short-term buffer
BUFFER_MAX_TURNS: int = int(os.environ.get("BUFFER_MAX_TURNS", "20"))

# Rate-limit sliding-window duration (seconds)
RATE_LIMIT_WINDOW_SECONDS: int = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Name for the barge-in Pub/Sub channel per session
BARGE_IN_CHANNEL_TEMPLATE = "barge_in:{session_id}"


# ────────────────────────────────────────────────────────────────────────────
# Key helpers
# ────────────────────────────────────────────────────────────────────────────

def _state_key(session_id: str) -> str:
    return f"session:{session_id}:state"


def _presence_key(session_id: str) -> str:
    return f"session:{session_id}:presence"


def _buffer_key(session_id: str) -> str:
    return f"session:{session_id}:buffer"


def _rate_limit_key(ip: str, endpoint: str) -> str:
    return f"rate_limit:{ip}:{endpoint}"


def barge_in_channel(session_id: str) -> str:
    """Return the Pub/Sub channel name for barge-in signals."""
    return BARGE_IN_CHANNEL_TEMPLATE.format(session_id=session_id)


# ────────────────────────────────────────────────────────────────────────────
# Client
# ────────────────────────────────────────────────────────────────────────────


class RedisClient:
    """Wrapper around ``redis.asyncio.Redis`` providing typed helpers for all
    per-session Redis operations.

    Usage::

        redis_client = RedisClient()
        await redis_client.connect()
        ...
        await redis_client.close()
    """

    def __init__(self, url: str = _REDIS_URL) -> None:
        self._url = url
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """Create the connection pool."""
        self._redis = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    @property
    def _r(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("RedisClient is not connected. Call await connect() first.")
        return self._redis

    # ------------------------------------------------------------------ #
    # Session state Hash                                                   #
    # ------------------------------------------------------------------ #

    async def set_turn_state(self, session_id: str, turn_state: str) -> None:
        """Set ``turn_state`` field (``generating`` | ``idle``) in session Hash."""
        key = _state_key(session_id)
        await self._r.hset(key, "turn_state", turn_state)
        await self._r.expire(key, SESSION_TTL_SECONDS)

    async def get_turn_state(self, session_id: str) -> str | None:
        """Return the current ``turn_state`` or *None* if the key does not exist."""
        return await self._r.hget(_state_key(session_id), "turn_state")

    async def delete_session_state(self, session_id: str) -> None:
        """Remove the session state Hash."""
        await self._r.delete(_state_key(session_id))

    # ------------------------------------------------------------------ #
    # WS presence Set                                                      #
    # ------------------------------------------------------------------ #

    async def mark_present(self, session_id: str) -> None:
        """Mark the WebSocket connection for *session_id* as active."""
        key = _presence_key(session_id)
        await self._r.set(key, "1", ex=SESSION_TTL_SECONDS)

    async def mark_absent(self, session_id: str) -> None:
        """Remove the WS presence flag for *session_id*."""
        await self._r.delete(_presence_key(session_id))

    async def is_present(self, session_id: str) -> bool:
        """Return *True* if there is an active WS connection for *session_id*."""
        value = await self._r.get(_presence_key(session_id))
        return value == "1"

    # ------------------------------------------------------------------ #
    # Short-term conversation buffer (List, capped)                        #
    # ------------------------------------------------------------------ #

    async def push_to_buffer(self, session_id: str, turn_json: str) -> None:
        """Append *turn_json* to the conversation buffer and keep it capped.

        Maintains at most ``BUFFER_MAX_TURNS`` entries (oldest entries are
        discarded when the cap is exceeded).
        """
        key = _buffer_key(session_id)
        pipe = self._r.pipeline()
        pipe.rpush(key, turn_json)
        pipe.ltrim(key, -BUFFER_MAX_TURNS, -1)
        pipe.expire(key, SESSION_TTL_SECONDS)
        await pipe.execute()

    async def get_buffer(self, session_id: str) -> list[str]:
        """Return all turns in the short-term buffer (oldest first)."""
        return await self._r.lrange(_buffer_key(session_id), 0, -1)

    async def clear_buffer(self, session_id: str) -> None:
        """Delete the conversation buffer for *session_id*."""
        await self._r.delete(_buffer_key(session_id))

    # ------------------------------------------------------------------ #
    # Barge-in Pub/Sub                                                     #
    # ------------------------------------------------------------------ #

    async def publish_barge_in(self, session_id: str) -> int:
        """Publish a barge-in cancellation signal.

        Returns the number of subscribers that received the message.
        """
        channel = barge_in_channel(session_id)
        return await self._r.publish(channel, "cancel")

    async def subscribe_barge_in(self, session_id: str) -> PubSub:
        """Return a *PubSub* object subscribed to the barge-in channel.

        The caller is responsible for unsubscribing and closing the PubSub
        when it is no longer needed.
        """
        pubsub = self._r.pubsub()
        await pubsub.subscribe(barge_in_channel(session_id))
        return pubsub

    # ------------------------------------------------------------------ #
    # Sliding-window rate limiting                                         #
    # ------------------------------------------------------------------ #

    async def increment_rate_limit(self, ip: str, endpoint: str) -> int:
        """Increment and return the request counter for *ip* + *endpoint*.

        The counter is reset after ``RATE_LIMIT_WINDOW_SECONDS`` seconds using
        a sliding-window approach backed by a Redis key with TTL.
        """
        key = _rate_limit_key(ip, endpoint)
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, RATE_LIMIT_WINDOW_SECONDS, nx=True)
        results = await pipe.execute()
        return int(results[0])

    async def increment_rate_limit_with_ttl(self, ip: str, endpoint: str) -> tuple[int, int]:
        """Increment counter and return ``(count, ttl)`` in a single pipeline call."""
        key = _rate_limit_key(ip, endpoint)
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, RATE_LIMIT_WINDOW_SECONDS, nx=True)
        pipe.ttl(key)
        results = await pipe.execute()
        count = int(results[0])
        ttl = max(int(results[2]), 0)
        return count, ttl

    async def get_rate_limit_count(self, ip: str, endpoint: str) -> int:
        """Return the current request count for *ip* + *endpoint*."""
        value = await self._r.get(_rate_limit_key(ip, endpoint))
        return int(value) if value else 0

    async def get_rate_limit_ttl(self, ip: str, endpoint: str) -> int:
        """Return the remaining TTL (seconds) for the rate-limit window."""
        ttl = await self._r.ttl(_rate_limit_key(ip, endpoint))
        return max(ttl, 0)

    # ------------------------------------------------------------------ #
    # Health probe                                                         #
    # ------------------------------------------------------------------ #

    async def ping(self) -> bool:
        """Return *True* if Redis responds to PING."""
        try:
            return bool(await self._r.ping())
        except Exception:
            return False
