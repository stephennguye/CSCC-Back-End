from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class Embedding:
    """A vectorized representation of a *Document* chunk stored in ChromaDB.

    Not persisted to PostgreSQL.  ``id`` is the ChromaDB document ID.
    """

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    chunk_text: str
    vector: list[float]
    metadata: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        *,
        document_id: uuid.UUID,
        chunk_index: int,
        chunk_text: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        embedding_id: uuid.UUID | None = None,
    ) -> Embedding:
        return cls(
            id=embedding_id or uuid.uuid4(),
            document_id=document_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            vector=vector,
            metadata=metadata,
        )
