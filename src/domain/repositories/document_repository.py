"""DocumentRepository — domain interface (Protocol).

Zero framework imports — pure Python typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid

    from src.domain.entities.document import Document


@runtime_checkable
class DocumentRepository(Protocol):
    """Persistence interface for *Document* records in PostgreSQL."""

    async def create(self, document: Document) -> Document:
        """Persist a new *Document* and return it."""
        ...  # pragma: no cover

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        """Return the *Document* for *document_id*, or *None* if not found."""
        ...  # pragma: no cover

    async def update_ingestion_status(
        self,
        document_id: uuid.UUID,
        *,
        status: str,
    ) -> None:
        """Update the ingestion lifecycle status of a *Document*."""
        ...  # pragma: no cover
