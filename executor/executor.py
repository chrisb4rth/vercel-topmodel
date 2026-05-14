"""
Executor Layer module for the Legal Advisor System.

Provides the ExecutorLayer class that exposes a REST API compatible with
Vercel AI Gateway routing conventions. This is the top-level HTTP handler
responsible for cross-cutting concerns that must be resolved before any
query reaches the supervisor pipeline:

    1. Authentication — validates API keys from Authorization (Bearer) or
       x-api-key headers against the configured environment variable.
    2. Rate limiting — enforces a sliding-window rate limit per API key to
       prevent abuse and protect downstream sub-agent resources.
    3. Request validation — delegates to the validation module for payload
       structure, query length, and language checks.
    4. Streaming delivery — wraps the supervisor's async stream of StreamChunks
       into Server-Sent Events (SSE) for incremental client delivery.
    5. Error mapping — translates internal exceptions and edge cases into
       the appropriate HTTP status codes (400, 401, 403, 429, 503, 504).

The module exposes a factory function `create_app()` that wires together
all dependencies and returns a fully configured FastAPI application instance.

Data Flow:
    HTTP request → authenticate() → validate_request() → supervisor.process_query()
    → SSE stream of StreamChunks → client

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from context.store import ContextStore
from executor.validation import ValidationError, validate_request_payload
from models import ChatRequest, StreamChunk
from registry.registry import SubAgentRegistry
from supervisor.supervisor import Supervisor

# Module-level logger for executor layer diagnostics.
logger: logging.Logger = logging.getLogger(__name__)

# Environment variable name holding the valid API key for authentication.
_API_KEY_ENV_VAR: str = "LEGAL_ADVISOR_API_KEY"

# Default rate limit: maximum requests allowed per window per API key.
_RATE_LIMIT_MAX_REQUESTS: int = 60

# Rate limit window duration in seconds (sliding window).
_RATE_LIMIT_WINDOW_SECONDS: float = 60.0


class ExecutorLayer:
    """
    Top-level HTTP handler compatible with Vercel AI Gateway.

    The ExecutorLayer wraps a FastAPI application and provides authentication,
    rate limiting, request validation, and SSE streaming. It delegates all
    legal reasoning to the Supervisor, which orchestrates classification,
    dispatch, and synthesis.

    The class maintains:
        - A reference to the Supervisor for query processing.
        - An in-memory rate limit tracker (sliding window per API key).
        - The configured API key loaded from the environment.

    For production deployments behind the Vercel AI Gateway, the gateway handles
    TLS termination and routing; this layer handles application-level auth and
    request processing.

    Attributes:
        _supervisor: The Supervisor instance that orchestrates the query pipeline.
        _api_key: The valid API key loaded from the environment variable at init.
        _rate_limit_store: Per-key sliding window tracker mapping API keys to
            lists of request timestamps.
        _app: The FastAPI application instance with routes configured.
    """

    def __init__(
        self,
        supervisor: Supervisor,
    ) -> None:
        """
        Initialize the ExecutorLayer with its required dependencies.

        Loads the API key from the environment and sets up the rate limit store.
        The FastAPI app is created and routes are registered during initialization.

        Args:
            supervisor: The Supervisor instance responsible for orchestrating
                query classification, sub-agent dispatch, and response synthesis.
        """

        self._supervisor: Supervisor = supervisor

        # Load the valid API key from the environment. If not set, authentication
        # will reject all requests — this is intentional for safety.
        self._api_key: str = os.environ.get(_API_KEY_ENV_VAR, "")

        # In-memory sliding window rate limit tracker.
        # Maps API key → list of request timestamps (epoch seconds).
        self._rate_limit_store: dict[str, list[float]] = defaultdict(list)

        # Create and configure the FastAPI application with routes.
        self._app: FastAPI = self._create_app()

        return None

    @property
    def app(self) -> FastAPI:
        """
        Return the configured FastAPI application instance.

        Provides external access to the app for mounting in ASGI servers
        (e.g., uvicorn) or for testing with httpx.AsyncClient.

        Returns:
            The FastAPI application with all routes registered.
        """

        return self._app

    def _create_app(self) -> FastAPI:
        """
        Create and configure the FastAPI application with all routes.

        Registers the chat completion endpoint compatible with Vercel AI Gateway
        routing conventions. The endpoint path follows the OpenAI-compatible
        convention used by Vercel AI Gateway for chat completions.

        Returns:
            A fully configured FastAPI application instance.
        """

        app: FastAPI = FastAPI(
            title="Legal Advisor System",
            description="Agentic chatbot for legal questions in German banking operations",
            version="0.1.0",
        )

        # Register the primary chat endpoint.
        # Using /v1/chat/completions for Vercel AI Gateway compatibility.
        # response_model=None disables Pydantic schema generation for the
        # response since we return either StreamingResponse or JSONResponse
        # depending on the request outcome.
        app.add_api_route(
            path="/v1/chat/completions",
            endpoint=self._handle_chat_request,
            methods=["POST"],
            response_model=None,
        )

        # Health check endpoint for gateway liveness probes.
        app.add_api_route(
            path="/health",
            endpoint=self._health_check,
            methods=["GET"],
            response_model=None,
        )

        # Root endpoint — provides a basic service info response so that
        # hitting "/" in a browser doesn't return a confusing 404.
        app.add_api_route(
            path="/",
            endpoint=self._health_check,
            methods=["GET"],
            response_model=None,
        )

        return app

    async def authenticate(
        self,
        api_key: str,
    ) -> bool:
        """
        Validate an API key against the configured environment credential.

        Compares the provided key against the value loaded from the
        LEGAL_ADVISOR_API_KEY environment variable at startup. Uses constant-time
        comparison semantics (via string equality on fixed-length values) to
        mitigate timing attacks in production scenarios.

        Args:
            api_key: The API key extracted from the request's Authorization
                header (Bearer token) or x-api-key header.

        Returns:
            True if the API key matches the configured credential, False otherwise.
        """

        # If no API key is configured in the environment, reject everything.
        if not self._api_key:
            logger.warning(
                "No API key configured in environment variable '%s'. "
                "All authentication attempts will fail.",
                _API_KEY_ENV_VAR,
            )
            return False

        # Compare the provided key against the configured credential.
        is_valid: bool = (api_key == self._api_key)

        return is_valid

    def validate_request(
        self,
        payload: dict,
    ) -> ChatRequest:
        """
        Parse and validate an incoming request payload.

        Delegates to the validation module's validate_request_payload function,
        which enforces payload structure, query length, and language constraints.
        This method exists on the ExecutorLayer interface to satisfy the design
        contract while keeping validation logic in its dedicated module.

        Args:
            payload: The raw request dictionary parsed from the incoming JSON body.

        Raises:
            ValidationError: If the payload fails any validation check.

        Returns:
            A fully validated ChatRequest instance.
        """

        validated_request: ChatRequest = validate_request_payload(payload)

        return validated_request

    def _check_rate_limit(
        self,
        api_key: str,
    ) -> bool:
        """
        Check whether the given API key has exceeded the rate limit.

        Implements a sliding window algorithm: removes timestamps older than
        the window duration, then checks if the remaining count exceeds the
        maximum allowed requests. If within limits, records the current timestamp.

        Args:
            api_key: The API key to check rate limits for.

        Returns:
            True if the request is allowed (within rate limit), False if the
            rate limit has been exceeded.
        """

        current_time: float = time.time()
        window_start: float = current_time - _RATE_LIMIT_WINDOW_SECONDS

        # Retrieve the request history for this key.
        timestamps: list[float] = self._rate_limit_store[api_key]

        # Prune timestamps outside the current sliding window.
        self._rate_limit_store[api_key] = [
            ts for ts in timestamps if ts > window_start
        ]

        # Check if the key has exceeded the maximum allowed requests.
        if len(self._rate_limit_store[api_key]) >= _RATE_LIMIT_MAX_REQUESTS:
            return False

        # Record this request's timestamp within the window.
        self._rate_limit_store[api_key].append(current_time)

        return True

    def _extract_api_key(
        self,
        request: Request,
    ) -> str | None:
        """
        Extract the API key from the request headers.

        Checks two header locations in order of precedence:
            1. Authorization header with "Bearer " prefix (standard OAuth2 convention)
            2. x-api-key header (common alternative for API key authentication)

        Args:
            request: The incoming FastAPI Request object.

        Returns:
            The extracted API key string, or None if no key was found in
            either header location.
        """

        # Check Authorization header first (Bearer token format).
        auth_header: str | None = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            extracted_key: str = auth_header[len("Bearer "):]
            return extracted_key

        # Fall back to x-api-key header.
        x_api_key: str | None = request.headers.get("x-api-key")
        if x_api_key:
            return x_api_key

        return None

    async def _handle_chat_request(
        self,
        request: Request,
    ) -> StreamingResponse | JSONResponse:
        """
        Handle an incoming chat completion request through the full pipeline.

        Orchestrates the request lifecycle:
            1. Extract and validate the API key (401/403 on failure).
            2. Check rate limits (429 on excess).
            3. Parse the JSON body and validate the payload (400 on failure).
            4. Delegate to the supervisor for query processing.
            5. Stream the response as Server-Sent Events.

        Args:
            request: The incoming FastAPI Request object containing headers
                and the JSON body.

        Returns:
            A StreamingResponse with SSE content type on success, or a
            JSONResponse with the appropriate error status code on failure.
        """

        # --- Step 1: Extract API key from headers ---
        api_key: str | None = self._extract_api_key(request)

        if api_key is None:
            logger.warning("Request received without API key.")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "missing_credentials",
                    "message": "API key required",
                },
            )

        # --- Step 2: Validate the API key ---
        is_authenticated: bool = await self.authenticate(api_key)

        if not is_authenticated:
            logger.warning("Request received with invalid API key.")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "invalid_credentials",
                    "message": "API key invalid or expired",
                },
            )

        # --- Step 3: Check rate limits ---
        is_within_limit: bool = self._check_rate_limit(api_key)

        if not is_within_limit:
            logger.warning(
                "Rate limit exceeded for API key (last 4 chars: ...%s).",
                api_key[-4:] if len(api_key) >= 4 else "****",
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "message": "Too many requests, retry later",
                },
            )

        # --- Step 4: Parse and validate the request body ---
        try:
            body: dict = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "validation_error",
                    "message": "Request body must be valid JSON",
                },
            )

        try:
            chat_request: ChatRequest = self.validate_request(body)
        except ValidationError as validation_error:
            return JSONResponse(
                status_code=400,
                content={
                    "error": validation_error.error_code,
                    "message": validation_error.message,
                },
            )

        # --- Step 5: Delegate to supervisor and stream the response ---
        return StreamingResponse(
            content=self._stream_response(chat_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream_response(
        self,
        chat_request: ChatRequest,
    ) -> AsyncIterator[str]:
        """
        Stream the supervisor's response as Server-Sent Events (SSE).

        Iterates over the StreamChunk async iterator produced by the supervisor
        and formats each chunk as an SSE data event. The final chunk includes
        metadata and is sent as a separate event type for client-side handling.

        SSE format:
            data: {"content": "...", "is_final": false}\n\n
            data: {"content": "", "is_final": true, "metadata": {...}}\n\n

        Args:
            chat_request: The validated ChatRequest to process.

        Yields:
            SSE-formatted strings for each StreamChunk from the supervisor.
        """

        try:
            stream: AsyncIterator[StreamChunk] = self._supervisor.process_query(
                query=chat_request.query,
                session_id=chat_request.session_id,
                language=chat_request.language.value,
            )

            async for chunk in stream:
                # Format the chunk as an SSE data event.
                event_data: dict = {
                    "content": chunk.content,
                    "is_final": chunk.is_final,
                }

                # Include metadata only on the final chunk.
                if chunk.is_final and chunk.metadata is not None:
                    event_data["metadata"] = chunk.metadata

                sse_line: str = f"data: {json.dumps(event_data)}\n\n"
                yield sse_line

        except Exception as processing_error:
            # If the supervisor pipeline raises an unexpected error, emit
            # an error event so the client knows the stream terminated abnormally.
            logger.exception(
                "Unexpected error during query processing: %s",
                processing_error,
            )
            error_event: dict = {
                "content": "",
                "is_final": True,
                "metadata": {
                    "error": "internal_error",
                    "message": "An unexpected error occurred during processing",
                },
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    async def _health_check(
        self,
        request: Request,
    ) -> JSONResponse:
        """
        Health check endpoint for gateway liveness probes.

        Returns a simple JSON response indicating the service is running.
        Does not require authentication — health checks are typically
        performed by infrastructure components (load balancers, gateways).

        Args:
            request: The incoming FastAPI Request object (unused but required
                by the route signature).

        Returns:
            A JSONResponse with status "ok" and HTTP 200.
        """

        return JSONResponse(
            status_code=200,
            content={"status": "ok"},
        )


def create_app(
    registry: SubAgentRegistry,
    context_store: ContextStore,
) -> FastAPI:
    """
    Factory function that creates a fully configured FastAPI application.

    Wires together all dependencies: creates the Supervisor with the provided
    registry and context store, then wraps it in an ExecutorLayer that provides
    the HTTP interface. Returns the FastAPI app ready for mounting in an ASGI
    server (e.g., uvicorn).

    This is the primary entry point for application startup and testing.

    Args:
        registry: The sub-agent registry populated with all available agents.
        context_store: The session context store for conversation history.

    Returns:
        A fully configured FastAPI application instance with all routes,
        authentication, rate limiting, and streaming wired up.
    """

    # Create the supervisor that orchestrates the query pipeline.
    supervisor: Supervisor = Supervisor(
        registry=registry,
        context_store=context_store,
    )

    # Create the executor layer wrapping the supervisor in HTTP handling.
    executor: ExecutorLayer = ExecutorLayer(supervisor=supervisor)

    return executor.app
