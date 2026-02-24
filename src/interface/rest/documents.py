"""POST /api/v1/documents/ingest — ingest a document into the knowledge base.

Flow:
  1. Validate the request body (content required, max 2 MB).
  2. Persist a ``Document`` record synchronously in PostgreSQL.
  3. Enqueue :class:`~src.application.use_cases.ingest_document.IngestDocumentUseCase`
     as a FastAPI ``BackgroundTask`` — the chunking and embedding happen
     asynchronously after the 202 response is returned.
  4. Return 202 Accepted with ``document_id`` and ``status: "ingesting"``.

See contracts/rest-api.md § POST /documents/ingest for the authoritative spec.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.interface.dtos.rest_responses import DocumentIngestResponse

logger = structlog.get_logger(__name__)

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["documents"])

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_CONTENT_BYTES: int = 2 * 1024 * 1024  # 2 MB

# ── Request model ─────────────────────────────────────────────────────────────


class DocumentIngestRequest(BaseModel):
    """Request body for POST /api/v1/documents/ingest."""

    model_config = ConfigDict(strict=True, extra="forbid")

    content: str = Field(..., description="Raw extracted text; max 2 MB")
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("content")
    @classmethod
    def _validate_content_size(cls, v: str) -> str:
        if len(v.encode("utf-8")) > _MAX_CONTENT_BYTES:
            raise ValueError("content exceeds maximum allowed size of 2 MB")
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


# ── Dependency injection ──────────────────────────────────────────────────────


def get_ingest_document(request: Request) -> Any:  # noqa: ANN401
    """FastAPI dependency — retrieve IngestDocumentUseCase from app.state."""
    use_case = getattr(request.app.state, "ingest_document", None)
    if use_case is None:
        raise RuntimeError(
            "IngestDocumentUseCase not initialised; app startup may have failed"
        )
    return use_case


def get_document_repository(request: Request) -> Any:  # noqa: ANN401
    """FastAPI dependency — retrieve PostgresDocumentRepository factory from app.state."""
    factory = getattr(request.app.state, "document_repo_factory", None)
    if factory is None:
        raise RuntimeError(
            "document_repo_factory not initialised; app startup may have failed"
        )
    return factory


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/documents/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentIngestResponse,
    summary="Ingest a document into the knowledge base",
    responses={
        400: {"description": "Invalid payload (missing content or content too large)"},
        415: {"description": "Unsupported media type"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def ingest_document(
    request: Request,
    body: DocumentIngestRequest,
    background_tasks: BackgroundTasks,
    ingest_document_uc: Any = Depends(get_ingest_document),  # noqa: ANN401, B008
    repo_factory: Any = Depends(get_document_repository),  # noqa: ANN401, B008
) -> DocumentIngestResponse:
    """Accept a document for asynchronous ingestion into the vector knowledge base.

    The document record is persisted immediately; embeddings are generated in a
    background task so the caller receives a 202 without waiting for the GPU.
    """
    from src.domain.entities.document import Document
    from src.domain.errors import PersistenceError
    from src.infrastructure.db.postgres.document_repo import PostgresDocumentRepository

    document_id = uuid.uuid4()
    now = datetime.now(UTC)

    document = Document(
        id=document_id,
        content=body.content,
        title=body.title,
        source=body.source,
        metadata={**(body.metadata or {}), "ingestion_status": "ingesting"},
        ingested_at=now,
    )

    # Persist synchronously via the repository factory
    try:
        async with repo_factory() as db_session, db_session.begin():
            doc_repo = PostgresDocumentRepository(db_session)
            await doc_repo.create(document)
    except PersistenceError as exc:
        logger.error("document_ingest_persist_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "PERSISTENCE_ERROR", "message": str(exc)},
        ) from exc

    logger.info(
        "document_ingest_accepted",
        document_id=str(document_id),
        title=body.title,
    )

    # Enqueue background ingestion (non-blocking)
    background_tasks.add_task(ingest_document_uc.execute, document_id)

    return DocumentIngestResponse(
        document_id=document_id,
        status="ingesting",
        message="Document accepted for processing. Embeddings will be generated asynchronously.",
    )
