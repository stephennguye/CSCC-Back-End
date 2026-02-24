"""IngestDocumentUseCase — background document ingestion pipeline.

Splits a raw text document into chunks, generates BAAI/bge-m3 embeddings for
each chunk, and upserts them to ChromaDB with full metadata.  Runs as a
FastAPI ``BackgroundTask`` — never blocks the request-response cycle.

Chunk parameters are configurable via environment variables:
  - ``RAG_CHUNK_SIZE``    : Target characters per chunk (default: 1000)
  - ``RAG_CHUNK_OVERLAP`` : Overlap between consecutive chunks (default: 100)
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import structlog

from src.domain.entities.embedding import Embedding
from src.domain.errors import PersistenceError

if TYPE_CHECKING:
    import uuid

    from src.domain.entities.document import Document
    from src.infrastructure.db.chroma.vector_repo import ChromaVectorRepository
    from src.infrastructure.db.postgres.document_repo import PostgresDocumentRepository
    from src.infrastructure.db.postgres.session import AsyncSessionFactory

logger = structlog.get_logger(__name__)

_CHUNK_SIZE: int = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
_CHUNK_OVERLAP: int = int(os.environ.get("RAG_CHUNK_OVERLAP", "100"))


# ─────────────────────────────────────────────────────────────────────────────
# Text chunking
# ─────────────────────────────────────────────────────────────────────────────


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping windows of *chunk_size* characters.

    Returns at least one chunk even for short texts.
    """
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Use case
# ─────────────────────────────────────────────────────────────────────────────


class IngestDocumentUseCase:
    """Chunk, embed, and upsert a document into the vector store.

    Injected via FastAPI DI by :func:`~src.main.create_app`.
    """

    def __init__(
        self,
        vector_repository: ChromaVectorRepository,
        session_factory: AsyncSessionFactory,
    ) -> None:
        self._vector_repo = vector_repository
        self._session_factory = session_factory

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def execute(self, document_id: uuid.UUID) -> None:
        """Run the full ingestion pipeline for *document_id*.

        Fetches the document text from PostgreSQL, splits it into chunks,
        generates embeddings, and upserts them to ChromaDB.  Updates the
        document's ingestion status in PostgreSQL on success or failure.

        This method is idempotent — re-running it replaces existing embeddings
        for the same *document_id* via ChromaDB upsert semantics.
        """
        from src.infrastructure.db.postgres.document_repo import (
            PostgresDocumentRepository,
        )

        async with self._session_factory() as db_session, db_session.begin():
            doc_repo = PostgresDocumentRepository(db_session)

            document = await doc_repo.get_by_id(document_id)
            if document is None:
                logger.error(
                    "ingest_document_not_found",
                    document_id=str(document_id),
                )
                return

            try:
                await self._ingest(document, doc_repo)
            except Exception as exc:
                logger.error(
                    "ingest_document_failed",
                    document_id=str(document_id),
                    error=str(exc),
                )
                with contextlib.suppress(PersistenceError):
                    await doc_repo.update_ingestion_status(
                        document_id, status="failed"
                    )

    async def _ingest(
        self,
        document: Document,
        doc_repo: PostgresDocumentRepository,
    ) -> None:
        """Core ingestion: split → embed → upsert → update status."""
        doc_id = document.id

        # 1. Split text into chunks
        chunks = _split_text(document.content, _CHUNK_SIZE, _CHUNK_OVERLAP)
        if not chunks:
            logger.warning("ingest_empty_document", document_id=str(doc_id))
            await doc_repo.update_ingestion_status(doc_id, status="complete")
            return

        logger.info(
            "ingest_chunks_split",
            document_id=str(doc_id),
            num_chunks=len(chunks),
        )

        # 2. Generate embeddings (batch call — offloaded to executor)
        vectors: list[list[float]] = await self._vector_repo.generate_embeddings(chunks)

        # 3. Build Embedding domain objects
        embeddings: list[Embedding] = []
        for chunk_index, (chunk_text, vector) in enumerate(
            zip(chunks, vectors, strict=True)
        ):
            embeddings.append(
                Embedding.create(
                    document_id=doc_id,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    vector=vector,
                    metadata={
                        "title": document.title or "",
                        "source": document.source or "",
                    },
                )
            )

        # 4. Upsert to ChromaDB
        await self._vector_repo.upsert_batch(embeddings)
        logger.info(
            "ingest_embeddings_upserted",
            document_id=str(doc_id),
            num_embeddings=len(embeddings),
        )

        # 5. Update ingestion status to complete
        await doc_repo.update_ingestion_status(doc_id, status="complete")
        logger.info("ingest_complete", document_id=str(doc_id))
