"""Tests for src.infrastructure.observability.circuit_breaker."""

from __future__ import annotations

import time

import pytest

from src.infrastructure.observability.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreakerInit:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=3, timeout=1.0)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_open is False

    def test_repr(self) -> None:
        cb = CircuitBreaker(name="my-service", failure_threshold=5, timeout=30.0)
        r = repr(cb)
        assert "my-service" in r
        assert "closed" in r
        assert "0/5" in r


class TestCircuitBreakerTransitions:
    async def test_success_keeps_closed(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=3, timeout=0.1)

        async def ok():
            return 42

        result = await cb.call(ok)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    async def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=3, timeout=10.0)

        async def fail():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        assert cb.state == CircuitState.OPEN

    async def test_open_rejects_immediately(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=1, timeout=10.0)

        async def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(fail)

        assert cb.state == CircuitState.OPEN

        async def ok():
            return 42

        with pytest.raises(CircuitOpenError):
            await cb.call(ok)

    async def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=1, timeout=0.1)

        async def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(fail)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        await _async_sleep(0.15)

        async def ok():
            return 42

        result = await cb.call(ok)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=1, timeout=0.1)

        async def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(fail)

        await _async_sleep(0.15)

        # Fail again during half-open → goes back to OPEN
        with pytest.raises(RuntimeError):
            await cb.call(fail)

        assert cb.state == CircuitState.OPEN

    async def test_context_manager_success(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=3, timeout=1.0)

        async with cb:
            result = 42

        assert result == 42
        assert cb.state == CircuitState.CLOSED

    async def test_failures_below_threshold_stay_closed(self) -> None:
        cb = CircuitBreaker(name="test", failure_threshold=5, timeout=1.0)

        async def fail():
            raise RuntimeError("boom")

        for _ in range(4):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        assert cb.state == CircuitState.CLOSED  # 4 < 5


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
