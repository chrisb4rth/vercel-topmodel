"""
Application entry point for the Legal Advisor System.

This module wires all system components together at startup and provides
the FastAPI application instance for both local development (via uvicorn)
and Vercel serverless deployment (via module-level `app` export).

Startup Sequence:
    1. Instantiate the SubAgentRegistry (empty at creation).
    2. Create concrete sub-agent instances (AccountSeizureAgent, InsolvencyAgent,
       SourceOfWealthAgent).
    3. Register each sub-agent in the registry — this populates the registry's
       internal metadata catalog so the supervisor's classifier can discover
       available domains at query time.
    4. Instantiate the ContextStore for per-session conversation history.
    5. Call the `create_app()` factory from the executor layer, passing the
       populated registry and context store. This internally creates the
       Supervisor (with classifier, dispatcher, synthesizer) and wraps it
       in the ExecutorLayer (authentication, rate limiting, streaming).
    6. Export the resulting FastAPI `app` at module level for Vercel deployment.
    7. For local development: run with uvicorn when executed as __main__.

The registry-based discovery pattern (Requirement 8.1, 8.2) means adding a
new sub-agent only requires importing it here and calling `registry.register()`
— no changes to the supervisor, classifier, or executor are needed.

Requirements: 6.1, 6.2, 8.1, 8.2
"""

import logging
import os

from fastapi import FastAPI

from agents.account_seizure import AccountSeizureAgent
from agents.insolvency import InsolvencyAgent
from agents.source_of_wealth import SourceOfWealthAgent
from context.store import ContextStore
from executor.executor import create_app
from registry.registry import SubAgentRegistry

# Configure module-level logger for startup diagnostics.
logger: logging.Logger = logging.getLogger(__name__)


async def _default_search_tool(query: str) -> list[dict]:
    """
    Placeholder search tool for the SourceOfWealthAgent.

    In production, this would call an external search API (e.g., Bing, Google,
    or an internal compliance database). For now, returns an empty result set
    so the agent can operate in a degraded mode, signaling that no public
    information was found and recommending enhanced due diligence.

    Replace this with a real implementation backed by httpx calls to a search
    API endpoint, configured via environment variables.

    Args:
        query: The search query string constructed by the SOW agent.

    Returns:
        A list of result dictionaries, each with "title", "snippet", and "url".
        Returns empty list in this placeholder implementation.
    """

    # TODO: Replace with actual search API integration (e.g., Bing Search API).
    # The API key would be loaded from os.environ["SEARCH_API_KEY"].
    logger.info("Search tool called with query: %s (placeholder — no results)", query)

    return []


def build_application() -> FastAPI:
    """
    Construct the fully wired FastAPI application with all components.

    Encapsulates the startup sequence in a function so it can be called
    from both the module-level initialization (for Vercel) and from tests
    that need a fresh application instance.

    The function performs registry-based sub-agent discovery at startup
    (Requirement 8.1): each sub-agent is instantiated and registered,
    making its metadata available to the supervisor's classifier without
    any hard-coded domain routing in the supervisor itself.

    Returns:
        A fully configured FastAPI application instance with authentication,
        rate limiting, streaming, and all sub-agents registered.
    """

    # Step 1: Create the sub-agent registry.
    registry: SubAgentRegistry = SubAgentRegistry()
    logger.info("SubAgentRegistry instantiated.")

    # Step 2: Instantiate concrete sub-agents.
    account_seizure_agent: AccountSeizureAgent = AccountSeizureAgent()
    insolvency_agent: InsolvencyAgent = InsolvencyAgent()
    source_of_wealth_agent: SourceOfWealthAgent = SourceOfWealthAgent(
        search_tool=_default_search_tool,
    )

    # Step 3: Register sub-agents — populates the registry's metadata catalog.
    # The registry enforces unique domain_ids, preventing accidental overwrites.
    registry.register(account_seizure_agent)
    logger.info(
        "Registered AccountSeizureAgent (domain_id='account_seizure')."
    )

    registry.register(insolvency_agent)
    logger.info(
        "Registered InsolvencyAgent (domain_id='insolvency')."
    )

    registry.register(source_of_wealth_agent)
    logger.info(
        "Registered SourceOfWealthAgent (domain_id='source_of_wealth')."
    )

    # Step 4: Create the per-session conversation context store.
    context_store: ContextStore = ContextStore()
    logger.info("ContextStore instantiated.")

    # Step 5: Build the FastAPI app via the executor factory.
    # This creates the Supervisor (classifier + dispatcher + synthesizer)
    # and wraps it in the ExecutorLayer (auth + rate limiting + SSE streaming).
    application: FastAPI = create_app(
        registry=registry,
        context_store=context_store,
    )
    logger.info(
        "Application built successfully. "
        "Registered %d sub-agent(s). "
        "Vercel AI Gateway compatible endpoint: POST /v1/chat/completions",
        len(registry.get_all_metadata()),
    )

    return application


# Module-level application instance — exported for Vercel serverless deployment.
# Vercel's Python runtime expects a module-level ASGI app variable.
app: FastAPI = build_application()


if __name__ == "__main__":
    import uvicorn

    # Configure basic logging for local development visibility.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Run the application locally with uvicorn.
    # Host 0.0.0.0 allows access from other devices on the network.
    # Port 8000 is the conventional default for FastAPI applications.
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000,
    )
