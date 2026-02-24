"""SQLAlchemy async ORM models for all five tables.

PII encryption:
  - ``Message.content``    : pgcrypto AES-256 via ``pgp_sym_encrypt`` / ``pgp_sym_decrypt``
  - ``Claim.student_name`` : same

The encryption key is read from the ``PGCRYPTO_KEY`` environment variable at
import time and used in the custom TypeDecorator for both columns.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime  # noqa: TC003
from typing import Any

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

# ────────────────────────────────────────────────────────────────────────────
# Pgcrypto key
# ────────────────────────────────────────────────────────────────────────────

_PGCRYPTO_KEY: str = os.environ.get("PGCRYPTO_KEY", "")


# ────────────────────────────────────────────────────────────────────────────
# Custom TypeDecorator for pgcrypto symmetric encryption
# ────────────────────────────────────────────────────────────────────────────


class PgpSymEncryptedText(TypeDecorator):
    """Stores plaintext via pgp_sym_encrypt and retrieves via pgp_sym_decrypt.

    Encryption and decryption happen in the PostgreSQL server; the Python layer
    sends and receives plaintext strings.  The key is taken from the
    ``PGCRYPTO_KEY`` environment variable.

    Usage::

        class MyModel(Base):
            secret: Mapped[str] = mapped_column(PgpSymEncryptedText())
    """

    impl = Text
    cache_ok = True

    def bind_expression(self, bindvalue: Any) -> Any:  # type: ignore[override]  # noqa: ANN401
        from sqlalchemy import cast, func
        from sqlalchemy.types import Text

        return func.pgp_sym_encrypt(bindvalue, cast(_PGCRYPTO_KEY, Text))

    def column_expression(self, col: Any) -> Any:  # type: ignore[override]  # noqa: ANN401
        from sqlalchemy import cast, func
        from sqlalchemy.dialects.postgresql import BYTEA

        return func.pgp_sym_decrypt(cast(col, BYTEA), _PGCRYPTO_KEY)


# ────────────────────────────────────────────────────────────────────────────
# Declarative base
# ────────────────────────────────────────────────────────────────────────────


class Base(AsyncAttrs, DeclarativeBase):
    pass


# ────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────


class CallSessionModel(Base):
    """Persisted representation of a *CallSession*."""

    __tablename__ = "call_session"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    def __repr__(self) -> str:
        return f"<CallSessionModel id={self.id} state={self.state}>"


class MessageModel(Base):
    """Persisted representation of a *Message* (single conversation turn).

    ``content`` is AES-256 encrypted via pgcrypto (FR-029).
    """

    __tablename__ = "message"
    __table_args__ = (
        Index("ix_message_session_sequence", "session_id", "sequence_number"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    content: Mapped[str] = mapped_column(PgpSymEncryptedText(), nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"<MessageModel id={self.id} role={self.role} seq={self.sequence_number}>"


class ClaimModel(Base):
    """Persisted representation of a post-call *Claim*.

    ``student_name`` is AES-256 encrypted via pgcrypto (FR-028).
    Unique constraint on ``session_id`` (at most one Claim per session).
    """

    __tablename__ = "claim"
    __table_args__ = (UniqueConstraint("session_id", name="uq_claim_session_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    student_name: Mapped[str | None] = mapped_column(
        PgpSymEncryptedText(), nullable=True
    )
    issue_category: Mapped[str | None] = mapped_column(Text, nullable=True)
    urgency_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    requested_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    follow_up_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    schema_version: Mapped[str] = mapped_column(
        String(16), nullable=False, default="v1"
    )

    def __repr__(self) -> str:
        return f"<ClaimModel id={self.id} session_id={self.session_id}>"


class ReminderModel(Base):
    """Persisted representation of an actionable *Reminder*."""

    __tablename__ = "reminder"
    __table_args__ = (
        Index("ix_reminder_session_id", "session_id"),
        Index("ix_reminder_target_due_at", "target_due_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    target_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    def __repr__(self) -> str:
        return f"<ReminderModel id={self.id} session_id={self.session_id}>"


class DocumentModel(Base):
    """Persisted representation of a knowledge-base *Document*."""

    __tablename__ = "document"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    def __repr__(self) -> str:
        return f"<DocumentModel id={self.id} title={self.title!r}>"
