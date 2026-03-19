"""Centralized FastAPI dependency functions.

All shared dependency injectors that retrieve infrastructure singletons
from ``app.state`` live here, eliminating duplication across router modules.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request, WebSocket


def get_session_factory(request: Request) -> Any:  # noqa: ANN401
    """FastAPI dependency -- retrieve the async session factory from app.state."""
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise RuntimeError("session_factory not initialised; app startup failed")
    return factory


def get_handle_call(request: Request) -> Any:  # noqa: ANN401
    """FastAPI dependency -- retrieve the HandleCallUseCase from app.state."""
    handle_call = getattr(request.app.state, "handle_call", None)
    if handle_call is None:
        raise RuntimeError("HandleCallUseCase not initialised; app startup may have failed")
    return handle_call


def get_handle_call_ws(websocket: WebSocket) -> Any:  # noqa: ANN401
    """FastAPI dependency -- retrieve the HandleCallUseCase from app.state (WebSocket variant)."""
    handle_call = getattr(websocket.app.state, "handle_call", None)
    if handle_call is None:
        raise RuntimeError("HandleCallUseCase not initialised; app startup may have failed")
    return handle_call


def get_tod_pipeline(request: Request) -> Any:  # noqa: ANN401
    """FastAPI dependency -- retrieve the TODPipelineUseCase from app.state."""
    pipeline = getattr(request.app.state, "tod_pipeline", None)
    if pipeline is None:
        raise RuntimeError("TODPipelineUseCase not initialised; app startup may have failed")
    return pipeline
