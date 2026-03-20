"""No-op OpenTelemetry stand-ins when tracing is not configured."""

from __future__ import annotations

from typing import Any


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None: ...  # noqa: ANN401
    def set_status(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401
    def record_exception(self, exc: BaseException) -> None: ...
    def __enter__(self) -> _NoopSpan: return self
    def __exit__(self, *args: Any) -> None: ...  # noqa: ANN401


class _NoopTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoopSpan:  # noqa: ANN401
        return _NoopSpan()
