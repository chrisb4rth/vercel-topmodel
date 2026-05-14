"""
Unit tests for the ExecutorLayer module.

Tests cover the HTTP-level behavior of the executor layer using httpx.AsyncClient
with FastAPI's test interface. Each test verifies a specific cross-cutting concern:
- Authentication: missing key (401), invalid key (403), valid key (passes)
- Rate limiting: excess requests (429)
- Request validation: malformed payloads (400)
- Streaming: SSE format compliance and response delivery
- Health check: liveness probe (200)

The tests use a minimal supervisor mock that yields predictable StreamChunks,
isolating the executor layer's HTTP handling from the full query pipeline.
"""

import json
import os
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from context.store import ContextStore
from executor.executor import ExecutorLayer, create_app
from models import StreamChunk
from registry.registry import SubAgentRegistry
from supervisor.supervisor import Supervisor


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# A known valid API key used across tests.
_TEST_API_KEY: str = "test-valid-api-key-12345"


@pytest.fixture
def registry() -> SubAgentRegistry:
    """Provide an empty SubAgentRegistry for executor tests."""

    return SubAgentRegistry()


@pytest.fixture
def context_store() -> ContextStore:
    """Provide a fresh ContextStore for executor tests."""

    return ContextStore()


@pytest.fixture
def supervisor(
    registry: SubAgentRegistry,
    context_store: ContextStore,
) -> Supervisor:
    """Provide a Supervisor instance wired with test dependencies."""

    return Supervisor(registry=registry, context_store=context_store)


@pytest.fixture
def executor(
    supervisor: Supervisor,
) -> ExecutorLayer:
    """Provide an ExecutorLayer with the test API key configured."""

    with patch.dict(os.environ, {"LEGAL_ADVISOR_API_KEY": _TEST_API_KEY}):
        executor_instance: ExecutorLayer = ExecutorLayer(supervisor=supervisor)

    return executor_instance


@pytest.fixture
def client(
    executor: ExecutorLayer,
) -> httpx.AsyncClient:
    """Provide an httpx.AsyncClient bound to the executor's FastAPI app."""

    from httpx import ASGITransport

    transport: ASGITransport = ASGITransport(app=executor.app)
    async_client: httpx.AsyncClient = httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    )

    return async_client


def _auth_headers(api_key: str = _TEST_API_KEY) -> dict[str, str]:
    """Helper to build Authorization headers with a Bearer token."""

    return {"Authorization": f"Bearer {api_key}"}


def _valid_payload() -> dict:
    """Helper to build a minimal valid request payload."""

    return {
        "query": "Was ist der Pfändungsfreibetrag?",
        "session_id": "test-session-001",
        "language": "de",
    }


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Tests for API key authentication behavior."""

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_401(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Request without any API key header receives HTTP 401."""

        response: httpx.Response = await client.post(
            "/v1/chat/completions",
            json=_valid_payload(),
        )

        assert response.status_code == 401
        body: dict = response.json()
        assert body["error"] == "missing_credentials"
        assert "API key required" in body["message"]

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_403(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Request with an incorrect API key receives HTTP 403."""

        response: httpx.Response = await client.post(
            "/v1/chat/completions",
            json=_valid_payload(),
            headers=_auth_headers("wrong-key-999"),
        )

        assert response.status_code == 403
        body: dict = response.json()
        assert body["error"] == "invalid_credentials"
        assert "invalid or expired" in body["message"]

    @pytest.mark.asyncio
    async def test_valid_bearer_token_passes_authentication(
        self,
        client: httpx.AsyncClient,
        executor: ExecutorLayer,
    ) -> None:
        """Request with a valid Bearer token passes the auth check."""

        # Mock the supervisor to avoid needing registered agents.
        async def mock_stream(*args, **kwargs) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Hello", is_final=True, metadata={"confidence": "high"})

        with patch.object(executor._supervisor, "process_query", side_effect=mock_stream):
            response: httpx.Response = await client.post(
                "/v1/chat/completions",
                json=_valid_payload(),
                headers=_auth_headers(_TEST_API_KEY),
            )

        # Should not be 401 or 403 — auth passed.
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_x_api_key_header_accepted(
        self,
        client: httpx.AsyncClient,
        executor: ExecutorLayer,
    ) -> None:
        """The x-api-key header is accepted as an alternative to Bearer."""

        async def mock_stream(*args, **kwargs) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="OK", is_final=True, metadata={})

        with patch.object(executor._supervisor, "process_query", side_effect=mock_stream):
            response: httpx.Response = await client.post(
                "/v1/chat/completions",
                json=_valid_payload(),
                headers={"x-api-key": _TEST_API_KEY},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_no_configured_api_key_rejects_all(
        self,
        supervisor: Supervisor,
    ) -> None:
        """When no API key is configured in env, all requests are rejected."""

        # Create executor without the env var set.
        with patch.dict(os.environ, {"LEGAL_ADVISOR_API_KEY": ""}, clear=False):
            executor_no_key: ExecutorLayer = ExecutorLayer(supervisor=supervisor)

        from httpx import ASGITransport
        transport: ASGITransport = ASGITransport(app=executor_no_key.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            response: httpx.Response = await test_client.post(
                "/v1/chat/completions",
                json=_valid_payload(),
                headers=_auth_headers("any-key"),
            )

        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(
        self,
        executor: ExecutorLayer,
    ) -> None:
        """Exceeding the rate limit returns HTTP 429."""

        from httpx import ASGITransport
        transport: ASGITransport = ASGITransport(app=executor.app)

        async def mock_stream(*args, **kwargs) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="OK", is_final=True, metadata={})

        # Fill up the rate limit window by injecting timestamps directly.
        # This avoids making 60 actual HTTP requests in the test.
        import time
        current_time: float = time.time()
        executor._rate_limit_store[_TEST_API_KEY] = [
            current_time - i for i in range(60)
        ]

        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            response: httpx.Response = await test_client.post(
                "/v1/chat/completions",
                json=_valid_payload(),
                headers=_auth_headers(_TEST_API_KEY),
            )

        assert response.status_code == 429
        body: dict = response.json()
        assert body["error"] == "rate_limited"
        assert "retry later" in body["message"]

    @pytest.mark.asyncio
    async def test_requests_within_limit_are_allowed(
        self,
        client: httpx.AsyncClient,
        executor: ExecutorLayer,
    ) -> None:
        """Requests within the rate limit window are processed normally."""

        async def mock_stream(*args, **kwargs) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="OK", is_final=True, metadata={})

        with patch.object(executor._supervisor, "process_query", side_effect=mock_stream):
            response: httpx.Response = await client.post(
                "/v1/chat/completions",
                json=_valid_payload(),
                headers=_auth_headers(_TEST_API_KEY),
            )

        # Should succeed — rate limit not exceeded.
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Request validation tests
# ---------------------------------------------------------------------------


class TestRequestValidation:
    """Tests for request payload validation at the HTTP level."""

    @pytest.mark.asyncio
    async def test_missing_query_field_returns_400(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Payload without 'query' field returns HTTP 400."""

        payload: dict = {"session_id": "s1", "language": "de"}

        response: httpx.Response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=_auth_headers(),
        )

        assert response.status_code == 400
        body: dict = response.json()
        assert body["error"] == "validation_error"
        assert "query" in body["message"]

    @pytest.mark.asyncio
    async def test_empty_query_returns_400(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Empty query string returns HTTP 400 with query_length_error."""

        payload: dict = {"query": "", "session_id": "s1", "language": "de"}

        response: httpx.Response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=_auth_headers(),
        )

        assert response.status_code == 400
        body: dict = response.json()
        assert body["error"] == "query_length_error"

    @pytest.mark.asyncio
    async def test_unsupported_language_returns_400(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Unsupported language code returns HTTP 400."""

        payload: dict = {"query": "Test", "session_id": "s1", "language": "fr"}

        response: httpx.Response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=_auth_headers(),
        )

        assert response.status_code == 400
        body: dict = response.json()
        assert body["error"] == "unsupported_language"

    @pytest.mark.asyncio
    async def test_invalid_json_body_returns_400(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Non-JSON request body returns HTTP 400."""

        response: httpx.Response = await client.post(
            "/v1/chat/completions",
            content=b"not valid json {{{",
            headers={
                **_auth_headers(),
                "content-type": "application/json",
            },
        )

        assert response.status_code == 400
        body: dict = response.json()
        assert body["error"] == "validation_error"


# ---------------------------------------------------------------------------
# Streaming response tests
# ---------------------------------------------------------------------------


class TestStreamingResponse:
    """Tests for SSE streaming format compliance."""

    @pytest.mark.asyncio
    async def test_response_is_sse_content_type(
        self,
        client: httpx.AsyncClient,
        executor: ExecutorLayer,
    ) -> None:
        """Successful responses use text/event-stream content type."""

        async def mock_stream(*args, **kwargs) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Answer text", is_final=True, metadata={"confidence": "high"})

        with patch.object(executor._supervisor, "process_query", side_effect=mock_stream):
            response: httpx.Response = await client.post(
                "/v1/chat/completions",
                json=_valid_payload(),
                headers=_auth_headers(),
            )

        assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_sse_chunks_are_valid_json(
        self,
        client: httpx.AsyncClient,
        executor: ExecutorLayer,
    ) -> None:
        """Each SSE data line contains valid JSON with expected fields."""

        async def mock_stream(*args, **kwargs) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Part 1", is_final=False)
            yield StreamChunk(content="", is_final=True, metadata={"confidence": "high"})

        with patch.object(executor._supervisor, "process_query", side_effect=mock_stream):
            response: httpx.Response = await client.post(
                "/v1/chat/completions",
                json=_valid_payload(),
                headers=_auth_headers(),
            )

        # Parse SSE lines from the response body.
        raw_body: str = response.text
        sse_lines: list[str] = [
            line for line in raw_body.split("\n") if line.startswith("data: ")
        ]

        assert len(sse_lines) == 2

        # First chunk: content with is_final=False.
        first_event: dict = json.loads(sse_lines[0].removeprefix("data: "))
        assert first_event["content"] == "Part 1"
        assert first_event["is_final"] is False

        # Final chunk: empty content with is_final=True and metadata.
        final_event: dict = json.loads(sse_lines[1].removeprefix("data: "))
        assert final_event["content"] == ""
        assert final_event["is_final"] is True
        assert final_event["metadata"]["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_sse_format_uses_data_prefix_and_double_newline(
        self,
        client: httpx.AsyncClient,
        executor: ExecutorLayer,
    ) -> None:
        """SSE events follow the 'data: ...\n\n' format convention."""

        async def mock_stream(*args, **kwargs) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Hello", is_final=True, metadata={})

        with patch.object(executor._supervisor, "process_query", side_effect=mock_stream):
            response: httpx.Response = await client.post(
                "/v1/chat/completions",
                json=_valid_payload(),
                headers=_auth_headers(),
            )

        # The raw body should contain "data: " prefix and end with double newline.
        raw_body: str = response.text
        assert "data: " in raw_body
        # Each event ends with \n\n (double newline separator).
        assert "\n\n" in raw_body


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Health check endpoint returns HTTP 200 with status ok."""

        response: httpx.Response = await client.get("/health")

        assert response.status_code == 200
        body: dict = response.json()
        assert body["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_does_not_require_auth(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Health check is accessible without authentication headers."""

        response: httpx.Response = await client.get("/health")

        # Should not be 401 or 403.
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_fastapi_instance(
        self,
        registry: SubAgentRegistry,
        context_store: ContextStore,
    ) -> None:
        """create_app returns a FastAPI application instance."""

        from fastapi import FastAPI

        with patch.dict(os.environ, {"LEGAL_ADVISOR_API_KEY": _TEST_API_KEY}):
            app = create_app(registry=registry, context_store=context_store)

        assert isinstance(app, FastAPI)
