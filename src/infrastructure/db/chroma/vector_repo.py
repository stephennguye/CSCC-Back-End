"""ChromaDB implementation of VectorRepository.

Embeds text using BAAI/bge-m3 (configurable via ``EMBEDDING_MODEL``) via
SentenceTransformer and persists/queries chunks in a ChromaDB collection.

Environment variables:
  - ``EMBEDDING_MODEL``  : SentenceTransformer model name (default: BAAI/bge-m3)
  - ``CHROMA_DB_PATH``  : Filesystem path for ChromaDB persistence (default: ./data/chroma)
  - ``RAG_TOP_K``        : Number of results to return by default (default: 5)

See contracts/rest-api.md and plan.md for design context.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.repositories.vector_repository import EmbeddingResult

if TYPE_CHECKING:
    from src.domain.entities.embedding import Embedding

logger = structlog.get_logger(__name__)

_EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
_CHROMA_DB_PATH: str = os.environ.get("CHROMA_DB_PATH", "./data/chroma")
_COLLECTION_NAME: str = "documents"
_DEFAULT_TOP_K: int = int(os.environ.get("RAG_TOP_K", "5"))


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singletons (loaded once, reused across calls)
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_encoder() -> Any:  # type: ignore[return]  # noqa: ANN401
    """Return the SentenceTransformer model (loaded once per process)."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    logger.info("loading_embedding_model", model=_EMBEDDING_MODEL)
    return SentenceTransformer(_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_chroma_client() -> Any:  # type: ignore[return]  # noqa: ANN401
    """Return the ChromaDB persistent client (opened once per process)."""
    import chromadb  # type: ignore[import-untyped]

    logger.info("opening_chromadb", path=_CHROMA_DB_PATH)
    return chromadb.PersistentClient(path=_CHROMA_DB_PATH)


def _get_collection(name: str = _COLLECTION_NAME) -> Any:  # type: ignore[return]  # noqa: ANN401
    """Return (or create) a ChromaDB collection by name."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _embed(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts and return their embedding vectors."""
    encoder = _get_encoder()
    vectors = encoder.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def _chroma_id(embedding_id: uuid.UUID) -> str:
    return str(embedding_id)


# ─────────────────────────────────────────────────────────────────────────────
# Repository implementation
# ─────────────────────────────────────────────────────────────────────────────


class ChromaVectorRepository:
    """Async-compatible ChromaDB-backed :class:`VectorRepository` implementation.

    Embedding generation is CPU/GPU-bound; it is offloaded to the default
    ``asyncio`` executor so the event loop is never blocked.
    """

    def __init__(self, collection_name: str = _COLLECTION_NAME) -> None:
        self._collection_name = collection_name

    # ------------------------------------------------------------------ #
    # Upsert                                                               #
    # ------------------------------------------------------------------ #

    async def upsert(self, embedding: Embedding) -> None:
        """Insert or update a single :class:`~src.domain.entities.embedding.Embedding`."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._sync_upsert_batch, [embedding]
        )

    async def upsert_batch(self, embeddings: list[Embedding]) -> None:
        """Batch upsert — more efficient than calling :meth:`upsert` in a loop."""
        if not embeddings:
            return
        await asyncio.get_event_loop().run_in_executor(
            None, self._sync_upsert_batch, embeddings
        )

    def _sync_upsert_batch(self, embeddings: list[Embedding]) -> None:
        """Synchronous batch upsert executed in the thread-pool executor."""
        collection = _get_collection(self._collection_name)

        ids: list[str] = []
        vectors: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for emb in embeddings:
            ids.append(_chroma_id(emb.id))
            vectors.append(emb.vector)
            documents.append(emb.chunk_text)
            meta: dict[str, Any] = {
                "document_id": str(emb.document_id),
                "chunk_index": emb.chunk_index,
            }
            if emb.metadata:
                meta.update(emb.metadata)
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("chroma_upsert_batch", count=len(ids))

    # ------------------------------------------------------------------ #
    # Similarity search                                                    #
    # ------------------------------------------------------------------ #

    async def similarity_search(
        self,
        query_text: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        collection: str | None = None,
    ) -> list[EmbeddingResult]:
        """Return the *top_k* most relevant chunks for *query_text*.

        Results are ranked by descending cosine similarity.
        Returns an empty list when the collection is empty or has no results.
        """
        col_name = collection or self._collection_name
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_search, query_text, top_k, col_name
        )

    def _sync_search(
        self, query_text: str, top_k: int, col_name: str
    ) -> list[EmbeddingResult]:
        """Synchronous similarity search executed in the thread-pool executor."""
        col = _get_collection(col_name)

        # Guard: ChromaDB raises if the collection is empty
        try:
            count = col.count()
        except Exception:
            count = 0
        if count == 0:
            return []

        # Embed query
        query_vectors = _embed([query_text])

        results = col.query(
            query_embeddings=query_vectors,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        output: list[EmbeddingResult] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_text, meta, distance in zip(docs, metas, distances, strict=False):
            # ChromaDB cosine distance = 1 - cosine_similarity when hnsw:space=cosine
            score = float(1.0 - distance)
            doc_id_str: str = meta.get("document_id", "")
            try:
                doc_id = uuid.UUID(doc_id_str)
            except ValueError:
                doc_id = uuid.uuid4()
            chunk_index: int = int(meta.get("chunk_index", 0))
            output.append(
                EmbeddingResult(
                    chunk_text=doc_text,
                    document_id=doc_id,
                    chunk_index=chunk_index,
                    score=score,
                    title=meta.get("title"),
                    metadata={
                        k: v
                        for k, v in meta.items()
                        if k not in {"document_id", "chunk_index", "title"}
                    },
                )
            )

        # Ensure descending order by score
        output.sort(key=lambda r: r.score, reverse=True)
        return output

    # ------------------------------------------------------------------ #
    # Delete                                                               #
    # ------------------------------------------------------------------ #

    async def delete_by_document_id(self, document_id: uuid.UUID) -> None:
        """Remove all embeddings associated with *document_id*."""
        doc_id_str = str(document_id)
        await asyncio.get_event_loop().run_in_executor(
            None, self._sync_delete, doc_id_str
        )

    def _sync_delete(self, doc_id_str: str) -> None:
        col = _get_collection(self._collection_name)
        col.delete(where={"document_id": doc_id_str})
        logger.debug("chroma_delete_by_document_id", document_id=doc_id_str)

    # ------------------------------------------------------------------ #
    # Embedding generation helper (exposed for IngestDocumentUseCase)     #
    # ------------------------------------------------------------------ #

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their float vectors."""
        return await asyncio.get_event_loop().run_in_executor(None, _embed, texts)
