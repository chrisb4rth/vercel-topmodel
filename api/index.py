"""
Vercel serverless function entry point.

This module serves as the handler for Vercel's Python serverless runtime.
Vercel expects a module at `api/index.py` that exports an ASGI-compatible
application instance named `app`.

Vercel's routing (configured in vercel.json) directs all incoming requests
to this handler, which then delegates to the FastAPI application's internal
routing (e.g., POST /v1/chat/completions, GET /health).

This separation keeps the application logic in `main.py` (usable for local
development with uvicorn) while providing the Vercel-specific entry point
here without duplicating wiring code.

Requirements: 6.1, 6.2
"""

import os
import sys

# Vercel's Python runtime uses the project root as the working directory,
# but sys.path may not include it for module resolution. Explicitly add
# the project root (parent of this file's directory) so that top-level
# packages like `agents`, `executor`, `supervisor`, etc. are importable.
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Re-export the application instance from the main entry point.
# Vercel's Python runtime will detect this as the ASGI handler.
from main import app  # noqa: E402

__all__: list[str] = ["app"]
