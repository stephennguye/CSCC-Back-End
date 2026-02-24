"""Initial schema: create all five tables with pgcrypto extension

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-02-24 00:00:00.000000+00:00

Tables created:
  - call_session
  - message          (content encrypted via pgcrypto)
  - claim            (student_name encrypted via pgcrypto; unique on session_id)
  - reminder
  - document

Indexes:
  - message(session_id, sequence_number)
  - claim(session_id)           — unique
  - reminder(session_id)
  - reminder(target_due_at)
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Activate pgcrypto extension ───────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ── call_session ─────────────────────────────────────────────────────
    op.create_table(
        "call_session",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "state",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ended_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
    )

    # ── message ──────────────────────────────────────────────────────────
    op.create_table(
        "message",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(8), nullable=False),
        # Stored as bytea from pgp_sym_encrypt — use Text to hold the ciphertext
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("sequence_number", sa.Integer, nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["call_session.id"],
            name="fk_message_session_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_message_session_sequence",
        "message",
        ["session_id", "sequence_number"],
    )

    # ── claim ─────────────────────────────────────────────────────────────
    op.create_table(
        "claim",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        # Stored as bytea from pgp_sym_encrypt
        sa.Column("student_name", sa.Text, nullable=True),
        sa.Column("issue_category", sa.Text, nullable=True),
        sa.Column("urgency_level", sa.String(16), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("requested_action", sa.Text, nullable=True),
        sa.Column("follow_up_date", sa.Date, nullable=True),
        sa.Column("extracted_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "schema_version",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'v1'"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["call_session.id"],
            name="fk_claim_session_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("session_id", name="uq_claim_session_id"),
    )
    op.create_index("ix_claim_session_id", "claim", ["session_id"])

    # ── reminder ──────────────────────────────────────────────────────────
    op.create_table(
        "reminder",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("target_due_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["call_session.id"],
            name="fk_reminder_session_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_reminder_session_id", "reminder", ["session_id"])
    op.create_index("ix_reminder_target_due_at", "reminder", ["target_due_at"])

    # ── document ──────────────────────────────────────────────────────────
    op.create_table(
        "document",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("source", sa.Text, nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "ingested_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("metadata", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("document")
    op.drop_index("ix_reminder_target_due_at", table_name="reminder")
    op.drop_index("ix_reminder_session_id", table_name="reminder")
    op.drop_table("reminder")
    op.drop_index("ix_claim_session_id", table_name="claim")
    op.drop_table("claim")
    op.drop_index("ix_message_session_sequence", table_name="message")
    op.drop_table("message")
    op.drop_table("call_session")
    op.execute("DROP EXTENSION IF EXISTS pgcrypto")
