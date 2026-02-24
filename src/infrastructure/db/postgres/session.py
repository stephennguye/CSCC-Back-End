"""Async SQLAlchemy session factory.

Provides a per-request ``AsyncSession`` via ``get_session()`` for use in
FastAPI dependency injection.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# ── Configuration ─────────────────────────────────────────────────────────────

_DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://cscc:cscc_pass@localhost:5432/cscc_db",
)

# ── Engine & session factory ──────────────────────────────────────────────────

engine = create_async_engine(
    _DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autobegin=True,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield a database session for one request scope."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
