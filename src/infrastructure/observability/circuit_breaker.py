"""Simple async circuit breaker implementation.

Supports three states:
- **CLOSED**  — normal operation; failures are counted.
- **OPEN**    — circuit tripped; calls fail immediately without attempting the
                underlying operation.
- **HALF_OPEN** — probe state; one call is allowed through to test recovery.

Configuration is via constructor args or environment variables:

``CIRCUIT_BREAKER_FAILURE_THRESHOLD``   — failures before opening (default: 5)
``CIRCUIT_BREAKER_TIMEOUT_SECONDS``     — seconds before half-open probe (default: 30)

Usage
-----
::

    breaker = CircuitBreaker(name="openai_llm")

    async def call_openai():
        async with breaker:
            # underlying async operation
            ...

    # Or use the decorator helper:
    result = await breaker.call(some_async_fn, *args, **kwargs)
"""

from __future__ import annotations

import asyncio
import enum
import os
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_FAILURE_THRESHOLD = int(os.environ.get("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
_DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("CIRCUIT_BREAKER_TIMEOUT_SECONDS", "30.0"))


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open."""


class CircuitBreaker:
    """Async-safe circuit breaker.

    Parameters
    ----------
    name:
        Human-readable name used in log messages and metrics.
    failure_threshold:
        Number of consecutive failures before the circuit opens.
    timeout:
        Seconds to wait in OPEN state before transitioning to HALF_OPEN.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._timeout = timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    # ------------------------------------------------------------------ #
    # Context manager API                                                  #
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> CircuitBreaker:
        await self._check_state()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        if exc_val is not None and not isinstance(exc_val, CircuitOpenError):
            await self._on_failure()
        else:
            await self._on_success()
        return False  # Do not suppress exceptions

    # ------------------------------------------------------------------ #
    # Callable helper                                                      #
    # ------------------------------------------------------------------ #

    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Execute *fn* guarded by this circuit breaker."""
        async with self:
            return await fn(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Internal state machine                                               #
    # ------------------------------------------------------------------ #

    async def _check_state(self) -> None:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._timeout:
                    logger.info(
                        "circuit_breaker_half_open",
                        name=self.name,
                        elapsed_seconds=round(elapsed, 2),
                    )
                    self._state = CircuitState.HALF_OPEN
                else:
                    logger.debug(
                        "circuit_breaker_open_rejecting",
                        name=self.name,
                        seconds_remaining=round(self._timeout - elapsed, 2),
                    )
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is OPEN. "
                        f"Retry after {self._timeout - elapsed:.1f}s."
                    )

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("circuit_breaker_closed", name=self.name)
                self._state = CircuitState.CLOSED
            self._failure_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN or (
                self._failure_count >= self._failure_threshold
            ):
                logger.warning(
                    "circuit_breaker_opened",
                    name=self.name,
                    failure_count=self._failure_count,
                )
                self._state = CircuitState.OPEN
                # Emit Prometheus metric
                import contextlib as _cl
                with _cl.suppress(Exception):
                    from src.infrastructure.observability.metrics import llm_errors_total
                    if "llm" in self.name.lower():
                        llm_errors_total.labels(
                            provider=self.name, error_type="circuit_open"
                        ).inc()

    # ------------------------------------------------------------------ #
    # repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self._state.value}, "
            f"failures={self._failure_count}/{self._failure_threshold})"
        )
