"""Alembic environment — async engine (asyncpg dialect).

This env.py supports both online (``alembic upgrade head``) and offline
(``alembic upgrade head --sql``) migration execution.

The ``DATABASE_URL`` environment variable must use the ``postgresql+asyncpg``
scheme.  Alembic automatically switches between asyncpg (async) and psycopg2
(offline SQL generation) based on the run context.
"""

from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

# Ensure the project root (containing the `src` package) is on sys.path so
# that `from src.infrastructure...` imports work when alembic is invoked from
# any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env so DATABASE_URL and other variables are available for local runs
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ── Alembic config ────────────────────────────────────────────────────────────

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Target metadata ───────────────────────────────────────────────────────────
# Import the ORM base so Alembic can introspect all registered models.

from src.infrastructure.db.postgres.models import Base  # noqa: E402

target_metadata = Base.metadata

# ── Database URL from environment ────────────────────────────────────────────

_DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/cscc",
)
config.set_main_option("sqlalchemy.url", _DATABASE_URL)


# ════════════════════════════════════════════════════════════════════════════
# Offline mode
# ════════════════════════════════════════════════════════════════════════════


def run_migrations_offline() -> None:
    """Emit DDL to stdout without connecting to the database."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ════════════════════════════════════════════════════════════════════════════
# Online mode (async)
# ════════════════════════════════════════════════════════════════════════════


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Connect via asyncpg and run migrations inside a sync context."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


# ────────────────────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
