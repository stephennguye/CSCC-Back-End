"""Shared fixtures for integration tests.

Uses the real PostgreSQL instance running in Docker (cscc_postgres).
Each test gets its own engine + transactional session.
"""

from __future__ import annotations

# Set PGCRYPTO_KEY BEFORE any model imports (models.py reads it at import time)
import os

if not os.environ.get("PGCRYPTO_KEY"):
    os.environ["PGCRYPTO_KEY"] = "test_key_for_integration_tests_only_32b"

import uuid
from typing import AsyncGenerator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
)

from src.domain.entities.call_session import CallSession
from src.infrastructure.db.postgres.models import Base

TEST_DATABASE_URL: str = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://cscc:cscc_pass@localhost:5432/cscc_db",
)


@pytest.fixture()
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional test session that rolls back after each test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()

    await engine.dispose()


@pytest.fixture()
def sample_session_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture()
def sample_call_session(sample_session_id: uuid.UUID) -> CallSession:
    return CallSession.create(session_id=sample_session_id)
