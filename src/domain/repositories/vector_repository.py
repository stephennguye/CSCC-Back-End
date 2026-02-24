"""VectorRepository — domain interface (Protocol).

Zero framework imports — pure Python typing only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid

    from src.domain.entities.embedding import Embedding


@dataclass
class EmbeddingResult:
    """A single result returned by a similarity search."""

    chunk_text: str
    document_id: uuid.UUID
    chunk_index: int
    score: float  # Cosine similarity; higher = more relevant
    title: str | None = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class VectorRepository(Protocol):
    """Vector search interface backed by ChromaDB."""

    async def upsert(self, embedding: Embedding) -> None:
        """Insert or update a document chunk embedding in ChromaDB."""
        ...  # pragma: no cover

    async def upsert_batch(self, embeddings: list[Embedding]) -> None:
        """Batch upsert multiple embeddings — more efficient than looping."""
        ...  # pragma: no cover

    async def similarity_search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
    ) -> list[EmbeddingResult]:
        """Return the *top_k* most relevant chunks for *query_text*.

        Results are ranked by descending similarity score.
        Returns an empty list (not an error) when the collection is empty or
        no relevant results exceed the minimum threshold.
        """
        ...  # pragma: no cover

    async def delete_by_document_id(self, document_id: uuid.UUID) -> None:
        """Remove all embeddings associated with *document_id*."""
        ...  # pragma: no cover
