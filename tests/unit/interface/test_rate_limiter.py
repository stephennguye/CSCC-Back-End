"""Tests for rate limiter middleware — path classification and IP extraction."""

from __future__ import annotations

from src.interface.middleware.rate_limiter import _client_ip, _get_limit_for_path


class TestGetLimitForPath:
    def test_health_exempt(self) -> None:
        assert _get_limit_for_path("/api/v1/health") is None

    def test_health_shortcut_exempt(self) -> None:
        assert _get_limit_for_path("/health") is None

    def test_metrics_exempt(self) -> None:
        assert _get_limit_for_path("/metrics") is None

    def test_docs_exempt(self) -> None:
        assert _get_limit_for_path("/docs") is None

    def test_redoc_exempt(self) -> None:
        assert _get_limit_for_path("/redoc") is None

    def test_openapi_exempt(self) -> None:
        assert _get_limit_for_path("/openapi.json") is None

    def test_websocket_exempt(self) -> None:
        assert _get_limit_for_path("/ws/calls/some-id") is None

    def test_sessions_path(self) -> None:
        result = _get_limit_for_path("/api/v1/sessions")
        assert result is not None
        limit, group = result
        assert group == "sessions"
        assert limit == 20  # default

    def test_conversations_path(self) -> None:
        result = _get_limit_for_path("/api/v1/conversations/abc/history")
        assert result is not None
        limit, group = result
        assert group == "conversations"

    def test_dialogue_path(self) -> None:
        result = _get_limit_for_path("/api/v1/dialogue/turn")
        assert result is not None
        limit, group = result
        assert group == "dialogue"

    def test_default_api_path(self) -> None:
        result = _get_limit_for_path("/api/v1/something-else")
        assert result is not None
        _, group = result
        assert group == "api_default"

    def test_non_api_path_returns_none(self) -> None:
        assert _get_limit_for_path("/some-random-path") is None


class TestClientIP:
    def test_x_forwarded_for_single(self) -> None:

        class FakeRequest:
            headers = {"X-Forwarded-For": "1.2.3.4"}
            client = None

        assert _client_ip(FakeRequest()) == "1.2.3.4"

    def test_x_forwarded_for_multiple(self) -> None:

        class FakeRequest:
            headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
            client = None

        assert _client_ip(FakeRequest()) == "1.2.3.4"

    def test_direct_client(self) -> None:

        class FakeClient:
            host = "10.0.0.1"

        class FakeRequest:
            headers = {}
            client = FakeClient()

        assert _client_ip(FakeRequest()) == "10.0.0.1"

    def test_no_client_returns_unknown(self) -> None:

        class FakeRequest:
            headers = {}
            client = None

        assert _client_ip(FakeRequest()) == "unknown"
