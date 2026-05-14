"""
Executor Layer package.

Exposes the REST API and manages cross-cutting concerns including
authentication, request validation, rate limiting, and streaming responses.
Compatible with Vercel AI Gateway routing conventions.
"""

from executor.executor import ExecutorLayer, create_app
from executor.validation import (
    ValidationError,
    validate_language,
    validate_query_length,
    validate_request_payload,
)
