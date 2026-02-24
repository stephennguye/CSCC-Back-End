from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Document:
    """A knowledge-base source record ingested via the document ingestion API."""

    id: uuid.UUID
    content: str
    ingested_at: datetime
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        *,
        content: str,
        title: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        document_id: uuid.UUID | None = None,
        ingested_at: datetime | None = None,
    ) -> Document:
        return cls(
            id=document_id or uuid.uuid4(),
            content=content,
            ingested_at=ingested_at or datetime.utcnow(),
            title=title,
            source=source,
            metadata=metadata,
        )
