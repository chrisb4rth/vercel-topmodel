"""
Vercel serverless function entry point.

This module serves as the handler for Vercel's Python serverless runtime.
Vercel expects a module at `api/index.py` that exports an ASGI-compatible
application instance. The module re-exports the fully wired FastAPI `app`
from the main application entry point.

Vercel's routing (configured in vercel.json) directs all incoming requests
to this handler, which then delegates to the FastAPI application's internal
routing (e.g., POST /v1/chat/completions, GET /health).

This separation keeps the application logic in `main.py` (usable for local
development with uvicorn) while providing the Vercel-specific entry point
here without duplicating wiring code.

Requirements: 6.1, 6.2
"""

# Re-export the application instance from the main entry point.
# Vercel's Python runtime will detect this as the ASGI handler.
from main import app

# Vercel expects the variable to be named `app` at module level.
# The import above satisfies this convention directly.
__all__: list[str] = ["app"]
