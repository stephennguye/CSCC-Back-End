"""Centralized API router registration.

Mounts all REST sub-routers under a consistent ``/api/v1`` prefix
and the WebSocket router at its own path.  Imported once by ``main.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

API_PREFIX = "/api/v1"


def _build_api_router() -> APIRouter:
    """Assemble all REST sub-routers into a single parent APIRouter."""
    from src.interface.rest.claims import router as claims_router
    from src.interface.rest.conversations import router as conversations_router
    from src.interface.rest.dialogue import router as dialogue_router
    from src.interface.rest.health import router as health_router
    from src.interface.rest.reminders import router as reminders_router
    from src.interface.rest.sessions import router as sessions_router

    api = APIRouter(prefix=API_PREFIX)
    api.include_router(sessions_router)
    api.include_router(dialogue_router)
    api.include_router(claims_router)
    api.include_router(reminders_router)
    api.include_router(conversations_router)
    api.include_router(health_router)
    return api


def register_routers(app: FastAPI) -> None:
    """Mount all application routers on *app*."""
    # REST endpoints under /api/v1
    app.include_router(_build_api_router())

    # WebSocket endpoint (no prefix — lives at /ws/calls/{session_id})
    from src.interface.ws.call_controller import router as ws_router

    app.include_router(ws_router)
