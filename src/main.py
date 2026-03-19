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


from fastapi import FastAPI, Request, Response

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
# Application factory
# ════════════════════════════════════════════════════════════════════════════

_log = structlog.get_logger(__name__)


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
            "Real-time Vietnamese AI call center backend: WebSocket audio streaming, "
            "Task-Oriented Dialogue pipeline (NLU → DST → Policy → NLG), "
            "structured claim extraction, and reminder generation."
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

    # Request-ID middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    app.add_middleware(BaseHTTPMiddleware, dispatch=_request_id_middleware)

    # Exception handlers (extracted to interface.exception_handlers)
    from src.interface.exception_handlers import register_exception_handlers

    register_exception_handlers(app)

    # Tracing (no-op when OTEL env var is absent)
    _configure_tracing(app)

    # Prometheus metrics (gracefully skipped if package not installed)
    _configure_metrics(app)

    # ── Routers (centralized in interface.api_router) ─────────────────────
    from src.interface.api_router import register_routers

    register_routers(app)

    # ── Rate limiting middleware (T058) ────────────────────────────────────
    from src.interface.middleware.rate_limiter import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware)

    return app


# ════════════════════════════════════════════════════════════════════════════
# Infrastructure startup / shutdown
# ════════════════════════════════════════════════════════════════════════════


async def _startup(app: FastAPI) -> None:
    """Initialise all infrastructure singletons and wire DI."""
    from src.application.use_cases.handle_call import HandleCallUseCase
    from src.application.use_cases.stream_conversation import StreamConversationUseCase
    from src.application.use_cases.tod_pipeline import TODPipelineUseCase
    from src.infrastructure.cache.redis_client import RedisClient
    from src.infrastructure.db.postgres.session import AsyncSessionFactory
    from src.infrastructure.dst.hybrid_dst_adapter import HybridDSTAdapter
    from src.infrastructure.nlg.template_nlg_adapter import TemplateNLGAdapter
    from src.infrastructure.nlu.jointbert_nlu_adapter import JointBERTNLUAdapter
    from src.infrastructure.policy.rule_policy_adapter import RulePolicyAdapter
    from src.infrastructure.stt.faster_whisper_adapter import FasterWhisperAdapter
    from src.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter
    from src.infrastructure.tts.gtts_adapter import GTTSAdapter

    # ── Infrastructure singletons ──────────────────────────────────────────
    redis = RedisClient()
    await redis.connect()
    app.state.redis = redis

    stt = FasterWhisperAdapter()
    tts_primary = EdgeTTSAdapter()
    tts_fallback = GTTSAdapter()

    # Pre-load the STT model at startup so the first call isn't delayed
    # by a ~50s model download.
    try:
        from src.infrastructure.stt.faster_whisper_adapter import _get_model
        _log.info("preloading_stt_model")
        await _get_model()
        _log.info("stt_model_preloaded")
    except Exception as exc:
        _log.warning("stt_model_preload_failed", error=str(exc))

    # ── TOD pipeline infrastructure ──────────────────────────────────────
    nlu_adapter = JointBERTNLUAdapter()
    dst_adapter = HybridDSTAdapter()
    policy_adapter = RulePolicyAdapter()
    nlg_adapter = TemplateNLGAdapter()

    tod_pipeline = TODPipelineUseCase(
        nlu=nlu_adapter,
        dst=dst_adapter,
        policy=policy_adapter,
        nlg=nlg_adapter,
    )
    app.state.tod_pipeline = tod_pipeline

    # ── Use cases ──────────────────────────────────────────────────────────
    stream_conversation = StreamConversationUseCase(
        stt=stt,
        tod_pipeline=tod_pipeline,
        tts_primary=tts_primary,
        tts_fallback=tts_fallback,
    )

    app.state.session_factory = AsyncSessionFactory

    handle_call = HandleCallUseCase(
        session_factory=AsyncSessionFactory,
        redis=redis,
        stream_conversation=stream_conversation,
        tod_pipeline=tod_pipeline,
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
