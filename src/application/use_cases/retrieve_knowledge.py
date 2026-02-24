"""RetrieveKnowledgeUseCase — RAG context retrieval.

Embeds the caller's query, performs a similarity search against the ChromaDB
vector store, and assembles a formatted context string that is injected into
the LLM system prompt for every conversation turn.

Key behaviours:
  - Returns empty string (not an error) when no relevant documents exist.
  - Filters chunks below ``RAG_RELEVANCE_THRESHOLD`` (default: 0.35).
  - Includes a circuit-breaker guard: on ChromaDB timeout or connection error
    the exception is caught, a structured warning is logged, and empty context
    is returned so the real-time call pipeline continues uninterrupted (T044).
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.domain.repositories.vector_repository import VectorRepository

logger = structlog.get_logger(__name__)

_RELEVANCE_THRESHOLD: float = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.35"))
_TOP_K: int = int(os.environ.get("RAG_TOP_K", "5"))
_CIRCUIT_TIMEOUT: float = float(os.environ.get("RAG_CIRCUIT_TIMEOUT", "3.0"))  # seconds


class RetrieveKnowledgeUseCase:
    """Query the vector store and return a ranked context string.

    Injected via FastAPI DI by :func:`~src.main.create_app`.
    """

    def __init__(self, vector_repository: VectorRepository) -> None:
        self._vector_repo = vector_repository

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def execute(self, query_text: str, *, top_k: int = _TOP_K) -> str:
        """Return a formatted context string for *query_text*.

        Args:
            query_text: The user's transcribed utterance to ground.
            top_k:      Maximum number of chunks to include in context.

        Returns:
            A newline-delimited context string ready for LLM injection, or an
            empty string when no relevant documents are found.  Never raises.
        """
        if not query_text.strip():
            return ""

        try:
            results = await asyncio.wait_for(
                self._vector_repo.similarity_search(query_text, top_k=top_k),
                timeout=_CIRCUIT_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "rag_circuit_breaker_timeout",
                query=query_text[:80],
                timeout=_CIRCUIT_TIMEOUT,
            )
            return ""
        except Exception as exc:
            logger.warning(
                "rag_circuit_breaker_error",
                query=query_text[:80],
                error=str(exc),
            )
            return ""

        # Filter below relevance threshold
        relevant = [r for r in results if r.score >= _RELEVANCE_THRESHOLD]
        if not relevant:
            return ""

        # Rank by descending score (already sorted by ChromaVectorRepository)
        chunks: list[str] = []
        for i, result in enumerate(relevant[:top_k], start=1):
            source_label = result.title or f"Document {result.document_id}"
            chunks.append(
                f"[{i}] Source: {source_label}\n{result.chunk_text.strip()}"
            )

        context = "\n\n".join(chunks)
        logger.debug(
            "rag_context_assembled",
            num_chunks=len(chunks),
            top_score=relevant[0].score,
        )
        return context
