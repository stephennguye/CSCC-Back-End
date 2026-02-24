"""PostgreSQL implementation of DocumentRepository.

Persists :class:`~src.domain.entities.document.Document` records using
SQLAlchemy async ORM.  Ingestion status is stored as a ``status`` key in the
``metadata`` JSONB column so no schema migration is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select

from src.domain.entities.document import Document
from src.domain.errors import PersistenceError
from src.infrastructure.db.postgres.models import DocumentModel

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


def _model_to_document(row: DocumentModel) -> Document:
    return Document(
        id=row.id,
        title=row.title,
        source=row.source,
        content=row.content,
        ingested_at=row.ingested_at,
        metadata=row.metadata_,
    )


class PostgresDocumentRepository:
    """Async PostgreSQL-backed :class:`DocumentRepository` implementation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # CRUD                                                                 #
    # ------------------------------------------------------------------ #

    async def create(self, document: Document) -> Document:
        """Persist a new *Document* record and return it."""
        try:
            row = DocumentModel(
                id=document.id,
                title=document.title,
                source=document.source,
                content=document.content,
                ingested_at=document.ingested_at,
                metadata_=document.metadata or {},
            )
            self._session.add(row)
            await self._session.flush()
            logger.debug("document_created", document_id=str(document.id))
            return _model_to_document(row)
        except Exception as exc:
            logger.error("document_create_failed", error=str(exc))
            raise PersistenceError(f"Failed to create document: {exc}") from exc

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        """Return the *Document* for *document_id*, or *None* if not found."""
        try:
            result = await self._session.execute(
                select(DocumentModel).where(DocumentModel.id == document_id)
            )
            row = result.scalars().first()
            if row is None:
                return None
            return _model_to_document(row)
        except Exception as exc:
            logger.error(
                "document_get_by_id_failed",
                document_id=str(document_id),
                error=str(exc),
            )
            raise PersistenceError(f"Failed to fetch document {document_id}: {exc}") from exc

    async def update_ingestion_status(
        self,
        document_id: uuid.UUID,
        *,
        status: str,
    ) -> None:
        """Update the ingestion lifecycle status stored in metadata.

        The ``status`` key is merged into the existing ``metadata`` JSONB column.
        Accepted values: ``"ingesting"``, ``"complete"``, ``"failed"``.
        """
        try:
            result = await self._session.execute(
                select(DocumentModel).where(DocumentModel.id == document_id)
            )
            row = result.scalars().first()
            if row is None:
                logger.warning(
                    "document_update_status_not_found",
                    document_id=str(document_id),
                    status=status,
                )
                return
            existing_meta: dict = dict(row.metadata_ or {})
            existing_meta["ingestion_status"] = status
            row.metadata_ = existing_meta
            await self._session.flush()
            logger.debug(
                "document_ingestion_status_updated",
                document_id=str(document_id),
                status=status,
            )
        except Exception as exc:
            logger.error(
                "document_update_status_failed",
                document_id=str(document_id),
                error=str(exc),
            )
            raise PersistenceError(
                f"Failed to update document {document_id} status: {exc}"
            ) from exc
