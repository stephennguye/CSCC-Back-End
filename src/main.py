"""FastAPI application factory.

Responsibilities
----------------
* Configure structlog for JSON structured logging.
* Bootstrap OpenTelemetry OTLP tracing with ``session_id`` context variable.
* Register PrometheusInstrumentApp middleware for /metrics scraping.
* Map domain error taxonomy to HTTP status codes via global exception handlers.
* Register API router placeholders (filled in by later phases).
* Expose ``create_app()`` factory for test isolation.
"""

from __future__ import annotations

import contextvars
import os
import uuid

# Load .env file for local development (no-op if python-dotenv is not installed
# or the file doesn't exist)
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(override=False)  # don't override vars already set in the environment
except ImportError:
    pass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from src.infrastructure.cache.redis_client import RedisClient


from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.domain.errors import (
    ClaimExtractionError,
    LLMFallbackExhaustedError,
    LLMTimeoutError,
    PayloadValidationError,
    PersistenceError,
    PromptInjectionDetectedError,
    RAGGroundingError,
    SessionAlreadyEndedError,
    SessionNotFoundError,
    TranscriptionError,
)

# ── Context variable for session_id span correlation ─────────────────────────

session_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)

# ════════════════════════════════════════════════════════════════════════════
# Logging (structlog)
# ════════════════════════════════════════════════════════════════════════════


def _configure_logging() -> None:
    """Configure structlog to emit JSON-formatted log records."""
    log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper()

    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level, logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# OpenTelemetry tracing
# ════════════════════════════════════════════════════════════════════════════


def _configure_tracing(app: FastAPI) -> None:
    """Bootstrap OTLP tracing when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set."""
    endpoint: str | None = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({SERVICE_NAME: "cscc-backend"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        structlog.get_logger().warning(
            "opentelemetry packages not available; tracing disabled"
        )


# ════════════════════════════════════════════════════════════════════════════
# Prometheus middleware
# ════════════════════════════════════════════════════════════════════════════


def _configure_metrics(app: FastAPI) -> None:
    """Register Prometheus instrumentation middleware and custom metric instruments."""
    import contextlib

    # Eagerly import our custom metrics so they register with the default registry
    with contextlib.suppress(Exception):
        import src.infrastructure.observability.metrics as _  # noqa: F401

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    except ImportError:
        structlog.get_logger().warning(
            "prometheus_fastapi_instrumentator not available; /metrics disabled"
        )


# ════════════════════════════════════════════════════════════════════════════
# Request ID / session_id injection middleware
# ════════════════════════════════════════════════════════════════════════════


async def _request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Inject request_id and optional session_id into structlog context."""
    request_id = str(uuid.uuid4())
    # Attempt to extract session_id from path or header
    session_id: str | None = request.path_params.get("session_id") or request.headers.get(
        "X-Session-Id"
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        session_id=session_id,
        method=request.method,
        path=request.url.path,
    )
    if session_id:
        session_id_ctx.set(session_id)

    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


# ════════════════════════════════════════════════════════════════════════════
# Domain error → HTTP status code mapping
# ════════════════════════════════════════════════════════════════════════════

_log = structlog.get_logger(__name__)


def _make_error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return a structured error envelope for Pydantic v2 validation failures.

        Converts FastAPI / Pydantic validation errors to the standard
        ``{"error": {"code": ..., "message": ..., "details": [...]}}`` envelope
        so clients always receive a consistent error shape (FR-023).
        """
        details = [
            {
                "field": ".".join(str(loc) for loc in e.get("loc", [])),
                "issue": e.get("msg", ""),
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "INVALID_PAYLOAD",
                    "message": "Request body validation failed.",
                    "details": details,
                }
            },
        )

    @app.exception_handler(SessionNotFoundError)
    async def handle_session_not_found(
        request: Request, exc: SessionNotFoundError
    ) -> JSONResponse:
        return _make_error_response("SESSION_NOT_FOUND", str(exc), status.HTTP_404_NOT_FOUND)

    @app.exception_handler(SessionAlreadyEndedError)
    async def handle_session_ended(
        request: Request, exc: SessionAlreadyEndedError
    ) -> JSONResponse:
        return _make_error_response(
            "SESSION_ENDED", str(exc), status.HTTP_409_CONFLICT
        )

    @app.exception_handler(PayloadValidationError)
    async def handle_payload_validation(
        request: Request, exc: PayloadValidationError
    ) -> JSONResponse:
        return _make_error_response(
            "INVALID_PAYLOAD", str(exc), status.HTTP_400_BAD_REQUEST
        )

    @app.exception_handler(PromptInjectionDetectedError)
    async def handle_prompt_injection(
        request: Request, exc: PromptInjectionDetectedError
    ) -> JSONResponse:
        _log.warning("prompt_injection_detected", detail=str(exc))
        return _make_error_response(
            "PROMPT_INJECTION_DETECTED",
            "Request blocked by security policy.",
            status.HTTP_400_BAD_REQUEST,
        )

    @app.exception_handler(LLMTimeoutError)
    async def handle_llm_timeout(
        request: Request, exc: LLMTimeoutError
    ) -> JSONResponse:
        return _make_error_response(
            "LLM_TIMEOUT", str(exc), status.HTTP_503_SERVICE_UNAVAILABLE
        )

    @app.exception_handler(LLMFallbackExhaustedError)
    async def handle_llm_fallback_exhausted(
        request: Request, exc: LLMFallbackExhaustedError
    ) -> JSONResponse:
        return _make_error_response(
            "LLM_FALLBACK_EXHAUSTED",
            "All LLM providers are currently unavailable.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(TranscriptionError)
    async def handle_transcription_error(
        request: Request, exc: TranscriptionError
    ) -> JSONResponse:
        return _make_error_response(
            "TRANSCRIPTION_ERROR", str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    @app.exception_handler(RAGGroundingError)
    async def handle_rag_grounding(
        request: Request, exc: RAGGroundingError
    ) -> JSONResponse:
        return _make_error_response(
            "RAG_GROUNDING_ERROR", str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    @app.exception_handler(ClaimExtractionError)
    async def handle_claim_extraction(
        request: Request, exc: ClaimExtractionError
    ) -> JSONResponse:
        return _make_error_response(
            "CLAIM_EXTRACTION_ERROR",
            str(exc),
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @app.exception_handler(PersistenceError)
    async def handle_persistence(
        request: Request, exc: PersistenceError
    ) -> JSONResponse:
        _log.error("persistence_error", detail=str(exc))
        return _make_error_response(
            "PERSISTENCE_ERROR",
            "A storage operation failed. Please try again.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ════════════════════════════════════════════════════════════════════════════
# Application factory
# ════════════════════════════════════════════════════════════════════════════


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    All sub-systems (logging, tracing, metrics, exception handlers, routers)
    are wired here.  Router implementations are populated in later phases.
    """
    _configure_logging()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup / shutdown: connect and disconnect infrastructure singletons."""
        await _startup(app)
        yield
        await _shutdown(app)

    app = FastAPI(
        title="AI Call Center Backend",
        description=(
            "Real-time AI-powered call center backend: WebSocket audio streaming, "
            "RAG-grounded LLM responses, structured claim extraction, and reminder generation."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow the dev frontend origin and any configured overrides
    from starlette.middleware.cors import CORSMiddleware

    cors_origins_env: str = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
    cors_origins: list[str] = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    app.add_middleware(BaseHTTPMiddleware, dispatch=_request_id_middleware)

    # Exception handlers
    _register_exception_handlers(app)

    # Tracing (no-op when OTEL env var is absent)
    _configure_tracing(app)

    # Prometheus metrics (gracefully skipped if package not installed)
    _configure_metrics(app)

    # ── Phase 3 Routers ───────────────────────────────────────────────────
    from src.interface.rest.documents import router as documents_router
    from src.interface.rest.sessions import router as sessions_router
    from src.interface.ws.call_controller import router as ws_router

    API_PREFIX = "/api/v1"  # noqa: N806
    app.include_router(sessions_router, prefix=API_PREFIX)
    app.include_router(documents_router, prefix=API_PREFIX)
    app.include_router(ws_router)

    # ── Phase 5 Routers ───────────────────────────────────────────────────────
    from src.interface.rest.claims import router as claims_router

    app.include_router(claims_router, prefix=API_PREFIX)

    # ── Phase 6 Routers ───────────────────────────────────────────────────
    from src.interface.rest.reminders import router as reminders_router

    app.include_router(reminders_router, prefix=API_PREFIX)

    # ── Phase 7 Routers ───────────────────────────────────────────────
    from src.interface.rest.conversations import router as conversations_router
    from src.interface.rest.health import router as health_router

    app.include_router(conversations_router, prefix=API_PREFIX)
    app.include_router(health_router, prefix=API_PREFIX)

    # ── Rate limiting middleware (T058) ────────────────────────────────────
    from src.interface.middleware.rate_limiter import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware)

    return app


# ════════════════════════════════════════════════════════════════════════════
# Infrastructure startup / shutdown
# ════════════════════════════════════════════════════════════════════════════


async def _startup(app: FastAPI) -> None:
    """Initialise all infrastructure singletons and wire DI."""
    from src.application.use_cases.extract_claims import ExtractClaimsUseCase
    from src.application.use_cases.generate_reminder import GenerateReminderUseCase
    from src.application.use_cases.handle_call import HandleCallUseCase
    from src.application.use_cases.ingest_document import IngestDocumentUseCase
    from src.application.use_cases.retrieve_knowledge import RetrieveKnowledgeUseCase
    from src.application.use_cases.stream_conversation import StreamConversationUseCase
    from src.domain.services.prompt_sanitizer import PromptSanitizer
    from src.infrastructure.cache.redis_client import RedisClient
    from src.infrastructure.db.chroma.vector_repo import ChromaVectorRepository
    from src.infrastructure.db.postgres.session import AsyncSessionFactory
    from src.infrastructure.llm.huggingface_client import HuggingFaceLLMAdapter
    from src.infrastructure.llm.openai_client import OpenAILLMAdapter
    from src.infrastructure.stt.faster_whisper_adapter import FasterWhisperAdapter
    from src.infrastructure.tts.coqui_tts_adapter import CoquiTTSAdapter
    from src.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter

    # ── Infrastructure singletons ──────────────────────────────────────────
    redis = RedisClient()
    await redis.connect()
    app.state.redis = redis

    stt = FasterWhisperAdapter()
    llm_primary = OpenAILLMAdapter()
    llm_fallback = HuggingFaceLLMAdapter()
    tts_primary = CoquiTTSAdapter()
    tts_fallback = EdgeTTSAdapter()
    sanitizer = PromptSanitizer()

    # ── Phase 4: RAG infrastructure ──────────────────────────────────────
    vector_repo = ChromaVectorRepository()
    retrieve_knowledge = RetrieveKnowledgeUseCase(vector_repository=vector_repo)
    ingest_document = IngestDocumentUseCase(
        vector_repository=vector_repo,
        session_factory=AsyncSessionFactory,
    )
    app.state.ingest_document = ingest_document
    app.state.document_repo_factory = AsyncSessionFactory

    # ── Use cases ──────────────────────────────────────────────────────────
    stream_conversation = StreamConversationUseCase(
        stt=stt,
        llm_primary=llm_primary,
        llm_fallback=llm_fallback,
        tts_primary=tts_primary,
        tts_fallback=tts_fallback,
        prompt_sanitizer=sanitizer,
        redis=redis,
        retrieve_knowledge=retrieve_knowledge,
    )

    # ── Phase 5: Claim extraction ────────────────────────────────────────
    extract_claims = ExtractClaimsUseCase(
        llm=llm_primary,
        session_factory=AsyncSessionFactory,
    )
    app.state.extract_claims = extract_claims

    # ── Phase 6: Reminder generation ────────────────────────────────────────
    generate_reminders = GenerateReminderUseCase(
        llm=llm_primary,
        session_factory=AsyncSessionFactory,
    )
    app.state.generate_reminders = generate_reminders
    app.state.session_factory = AsyncSessionFactory

    handle_call = HandleCallUseCase(
        session_factory=AsyncSessionFactory,
        redis=redis,
        stream_conversation=stream_conversation,
        extract_claims=extract_claims,
        generate_reminders=generate_reminders,
    )
    app.state.handle_call = handle_call

    _log.info("infrastructure_startup_complete")


async def _shutdown(app: FastAPI) -> None:
    """Gracefully disconnect infrastructure singletons."""

    redis: RedisClient | None = getattr(app.state, "redis", None)
    if redis is not None:
        await redis.close()
    _log.info("infrastructure_shutdown_complete")


# ────────────────────────────────────────────────────────────────────────────
# ASGI entry point
# ────────────────────────────────────────────────────────────────────────────

app: FastAPI = create_app()
