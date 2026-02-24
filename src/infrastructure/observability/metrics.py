"""Prometheus instrumentation for the AI Call Center Backend.

Defines explicit Counter and Histogram instruments for:
- WebSocket connection lifecycle (connected, disconnected, barge-in)
- LLM call duration and fallback activations
- STT / TTS processing duration
- REST endpoint request latency (supplement to prometheus_fastapi_instrumentator)

Usage
-----
Import and call the helpers directly from infrastructure / use-case modules.
The metrics are auto-registered with the default Prometheus registry on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Histogram  # type: ignore[import-untyped]


def _make_counter(name: str, doc: str, labels: list[str] | None = None):  # type: ignore[return]  # noqa: ANN202
    try:
        from prometheus_client import Counter  # type: ignore[import-untyped]

        return Counter(name, doc, labels or [])
    except ImportError:
        return _NoopMetric()


def _make_histogram(  # noqa: ANN202
    name: str,
    doc: str,
    labels: list[str] | None = None,
    buckets: tuple[float, ...] | None = None,
):
    try:
        from prometheus_client import Histogram  # type: ignore[import-untyped]

        kwargs = {}
        if buckets:
            kwargs["buckets"] = buckets
        return Histogram(name, doc, labels or [], **kwargs)
    except ImportError:
        return _NoopMetric()


# ────────────────────────────────────────────────────────────────────────────
# No-op fallback (prometheus_client not installed)
# ────────────────────────────────────────────────────────────────────────────


class _NoopMetric:
    """Drop-in replacement for Prometheus metrics when the library is absent."""

    def labels(self, **_kwargs) -> _NoopMetric:  # noqa: ANN003
        return self

    def inc(self, _amount: float = 1) -> None:
        pass

    def observe(self, _value: float) -> None:
        pass

    def time(self) -> object:
        import contextlib

        return contextlib.nullcontext()


# ════════════════════════════════════════════════════════════════════════════
# WebSocket — connection lifecycle
# ════════════════════════════════════════════════════════════════════════════

ws_connections_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_ws_connections_total",
    "Total number of WebSocket call sessions initiated.",
)

ws_disconnections_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_ws_disconnections_total",
    "Total number of WebSocket disconnections (clean or abrupt).",
    labels=["reason"],  # "clean" | "abrupt" | "error"
)

ws_bargein_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_ws_bargein_total",
    "Total barge-in events (caller speaking during AI response).",
)

# ════════════════════════════════════════════════════════════════════════════
# LLM — latency and fallback
# ════════════════════════════════════════════════════════════════════════════

llm_request_duration_seconds: Histogram = _make_histogram(  # type: ignore[type-arg]
    "cscc_llm_request_duration_seconds",
    "Time in seconds from first LLM token request to last token received.",
    labels=["provider"],  # "openai" | "huggingface"
    buckets=(0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, float("inf")),
)

llm_fallback_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_llm_fallback_total",
    "Total LLM fallback activations (primary failed, using fallback provider).",
    labels=["from_provider", "to_provider"],
)

llm_errors_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_llm_errors_total",
    "Total LLM errors (timeout, fallback exhausted, etc.).",
    labels=["provider", "error_type"],
)

# ════════════════════════════════════════════════════════════════════════════
# STT — transcription processing time
# ════════════════════════════════════════════════════════════════════════════

stt_processing_duration_seconds: Histogram = _make_histogram(  # type: ignore[type-arg]
    "cscc_stt_processing_duration_seconds",
    "Time in seconds to transcribe one audio segment.",
    labels=["provider"],  # "faster_whisper"
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")),
)

stt_errors_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_stt_errors_total",
    "Total STT transcription errors.",
    labels=["provider"],
)

# ════════════════════════════════════════════════════════════════════════════
# TTS — synthesis processing time
# ════════════════════════════════════════════════════════════════════════════

tts_processing_duration_seconds: Histogram = _make_histogram(  # type: ignore[type-arg]
    "cscc_tts_processing_duration_seconds",
    "Time in seconds to synthesize one TTS response.",
    labels=["provider"],  # "coqui" | "edge_tts"
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")),
)

tts_fallback_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_tts_fallback_total",
    "Total TTS fallback activations (Coqui failed, using edge-tts).",
)

tts_errors_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_tts_errors_total",
    "Total TTS synthesis errors.",
    labels=["provider"],
)

# ════════════════════════════════════════════════════════════════════════════
# REST — additional per-route latency (complements prometheus_fastapi_instrumentator)
# ════════════════════════════════════════════════════════════════════════════

rest_request_duration_seconds: Histogram = _make_histogram(  # type: ignore[type-arg]
    "cscc_rest_request_duration_seconds",
    "REST endpoint request latency in seconds.",
    labels=["method", "route", "status_code"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")),
)

rest_requests_total: Counter = _make_counter(  # type: ignore[type-arg]
    "cscc_rest_requests_total",
    "Total REST requests handled.",
    labels=["method", "route", "status_code"],
)
