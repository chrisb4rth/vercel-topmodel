"""
Unit tests for the ParallelDispatcher.

Covers the four key behaviors:
    1. Successful dispatch — all agents respond within timeout.
    2. Timeout handling — an agent exceeding the timeout produces timed_out=True.
    3. Error handling — an agent raising an exception produces an error result.
    4. Parallel execution — multiple slow agents run concurrently, not sequentially.

Each test uses lightweight mock agents that extend BaseSubAgent to control
timing and response behavior without external dependencies.
"""

import asyncio
import time

import pytest

from agents.base import BaseSubAgent
from models import (
    ConfidenceLevel,
    ConversationContext,
    LegalReference,
    SubAgentMetadata,
    SubAgentResponse,
)
from supervisor.dispatcher import ParallelDispatcher


# ---------------------------------------------------------------------------
# Test fixtures: mock sub-agents with controllable behavior
# ---------------------------------------------------------------------------


class SuccessAgent(BaseSubAgent):
    """Agent that immediately returns a successful response."""

    def __init__(
        self,
        domain_id: str = "test_domain",
        delay: float = 0.0,
    ) -> None:
        self._domain_id: str = domain_id
        self._delay: float = delay

    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """Return a canned response after an optional delay."""

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        return SubAgentResponse(
            domain_id=self._domain_id,
            answer_body=f"Answer for: {query}",
            references=[
                LegalReference(law_name="ZPO", paragraph="§ 850c"),
            ],
            confidence=ConfidenceLevel.HIGH,
        )

    def get_metadata(self) -> SubAgentMetadata:
        """Return metadata identifying this agent's domain."""

        return SubAgentMetadata(
            domain_id=self._domain_id,
            description="Test agent",
            supported_categories=["test"],
        )


class SlowAgent(BaseSubAgent):
    """Agent that sleeps longer than the timeout to trigger TimeoutError."""

    def __init__(
        self,
        domain_id: str = "slow_domain",
        delay: float = 60.0,
    ) -> None:
        self._domain_id: str = domain_id
        self._delay: float = delay

    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """Sleep beyond the timeout threshold."""

        await asyncio.sleep(self._delay)
        # This line is unreachable when timeout is enforced
        return SubAgentResponse(
            domain_id=self._domain_id,
            answer_body="Should not reach here",
            references=[],
            confidence=ConfidenceLevel.LOW,
        )

    def get_metadata(self) -> SubAgentMetadata:
        """Return metadata identifying this agent's domain."""

        return SubAgentMetadata(
            domain_id=self._domain_id,
            description="Slow test agent",
            supported_categories=["slow"],
        )


class ErrorAgent(BaseSubAgent):
    """Agent that raises an exception during query handling."""

    def __init__(
        self,
        domain_id: str = "error_domain",
        error_message: str = "Something went wrong",
    ) -> None:
        self._domain_id: str = domain_id
        self._error_message: str = error_message

    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """Raise a RuntimeError to simulate agent failure."""

        raise RuntimeError(self._error_message)

    def get_metadata(self) -> SubAgentMetadata:
        """Return metadata identifying this agent's domain."""

        return SubAgentMetadata(
            domain_id=self._domain_id,
            description="Error test agent",
            supported_categories=["error"],
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dispatcher() -> ParallelDispatcher:
    """Create a fresh ParallelDispatcher instance."""

    return ParallelDispatcher()


@pytest.fixture
def empty_context() -> ConversationContext:
    """Create an empty conversation context for testing."""

    return ConversationContext(exchanges=[])


# ---------------------------------------------------------------------------
# Test: Successful dispatch
# ---------------------------------------------------------------------------


class TestSuccessfulDispatch:
    """Verify that agents returning normally produce success results."""

    async def test_single_agent_success(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """A single agent responding successfully yields one result with response set."""

        agent: SuccessAgent = SuccessAgent(domain_id="account_seizure")

        results = await dispatcher.dispatch(
            agents=[agent],
            query="What is the protected amount?",
            context=empty_context,
        )

        assert len(results) == 1
        assert results[0].domain_id == "account_seizure"
        assert results[0].response is not None
        assert results[0].timed_out is False
        assert results[0].error is None
        assert results[0].response.answer_body == "Answer for: What is the protected amount?"

    async def test_multiple_agents_success(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """Multiple agents all succeeding yields results in input order."""

        agents: list[BaseSubAgent] = [
            SuccessAgent(domain_id="account_seizure"),
            SuccessAgent(domain_id="insolvency"),
        ]

        results = await dispatcher.dispatch(
            agents=agents,
            query="Priority of claims in insolvency",
            context=empty_context,
        )

        assert len(results) == 2
        assert results[0].domain_id == "account_seizure"
        assert results[1].domain_id == "insolvency"
        assert all(r.response is not None for r in results)
        assert all(r.timed_out is False for r in results)
        assert all(r.error is None for r in results)

    async def test_empty_agent_list(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """Dispatching to zero agents returns an empty result list."""

        results = await dispatcher.dispatch(
            agents=[],
            query="Any query",
            context=empty_context,
        )

        assert results == []


# ---------------------------------------------------------------------------
# Test: Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Verify that agents exceeding the timeout produce timed_out results."""

    async def test_agent_timeout(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """An agent sleeping beyond the timeout yields timed_out=True."""

        # Use a very short timeout so the test runs quickly
        agent: SlowAgent = SlowAgent(domain_id="slow_domain", delay=5.0)

        results = await dispatcher.dispatch(
            agents=[agent],
            query="Will this time out?",
            context=empty_context,
            timeout_seconds=0.1,
        )

        assert len(results) == 1
        assert results[0].domain_id == "slow_domain"
        assert results[0].timed_out is True
        assert results[0].response is None
        assert results[0].error is None

    async def test_mixed_success_and_timeout(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """When one agent succeeds and another times out, both results are correct."""

        agents: list[BaseSubAgent] = [
            SuccessAgent(domain_id="fast_domain"),
            SlowAgent(domain_id="slow_domain", delay=5.0),
        ]

        results = await dispatcher.dispatch(
            agents=agents,
            query="Mixed results query",
            context=empty_context,
            timeout_seconds=0.1,
        )

        assert len(results) == 2
        # Fast agent succeeded
        assert results[0].domain_id == "fast_domain"
        assert results[0].response is not None
        assert results[0].timed_out is False
        # Slow agent timed out
        assert results[1].domain_id == "slow_domain"
        assert results[1].timed_out is True
        assert results[1].response is None


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify that agents raising exceptions produce error results."""

    async def test_agent_error(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """An agent raising an exception yields an error result with the message."""

        agent: ErrorAgent = ErrorAgent(
            domain_id="error_domain",
            error_message="Knowledge base unavailable",
        )

        results = await dispatcher.dispatch(
            agents=[agent],
            query="This will fail",
            context=empty_context,
        )

        assert len(results) == 1
        assert results[0].domain_id == "error_domain"
        assert results[0].error == "Knowledge base unavailable"
        assert results[0].response is None
        assert results[0].timed_out is False

    async def test_mixed_success_and_error(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """When one agent succeeds and another errors, both results are correct."""

        agents: list[BaseSubAgent] = [
            SuccessAgent(domain_id="good_domain"),
            ErrorAgent(domain_id="bad_domain", error_message="Crash"),
        ]

        results = await dispatcher.dispatch(
            agents=agents,
            query="Partial failure",
            context=empty_context,
        )

        assert len(results) == 2
        assert results[0].response is not None
        assert results[0].error is None
        assert results[1].response is None
        assert results[1].error == "Crash"

    async def test_all_agents_fail(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """When all agents fail (mix of timeout and error), all results reflect failure."""

        agents: list[BaseSubAgent] = [
            SlowAgent(domain_id="timeout_domain", delay=5.0),
            ErrorAgent(domain_id="error_domain", error_message="Broken"),
        ]

        results = await dispatcher.dispatch(
            agents=agents,
            query="Total failure",
            context=empty_context,
            timeout_seconds=0.1,
        )

        assert len(results) == 2
        assert results[0].timed_out is True
        assert results[1].error == "Broken"


# ---------------------------------------------------------------------------
# Test: Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    """Verify that agents run concurrently, not sequentially."""

    async def test_parallel_timing(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """
        Three agents each sleeping 0.2s should complete in ~0.2s total (parallel),
        not ~0.6s (sequential). We allow up to 0.5s to account for overhead.
        """

        agents: list[BaseSubAgent] = [
            SuccessAgent(domain_id="agent_a", delay=0.2),
            SuccessAgent(domain_id="agent_b", delay=0.2),
            SuccessAgent(domain_id="agent_c", delay=0.2),
        ]

        start_time: float = time.monotonic()

        results = await dispatcher.dispatch(
            agents=agents,
            query="Parallel test",
            context=empty_context,
            timeout_seconds=5.0,
        )

        elapsed: float = time.monotonic() - start_time

        # All agents should succeed
        assert len(results) == 3
        assert all(r.response is not None for r in results)

        # Total time should be well under 0.6s (sequential would be ~0.6s)
        # Allow generous margin for CI environments
        assert elapsed < 0.5, (
            f"Expected parallel execution under 0.5s, but took {elapsed:.3f}s"
        )

    async def test_result_order_matches_input_order(
        self,
        dispatcher: ParallelDispatcher,
        empty_context: ConversationContext,
    ) -> None:
        """
        Results are returned in the same order as the input agents list,
        regardless of which agent finishes first.
        """

        # Agent B finishes faster than Agent A, but results should still
        # be ordered [A, B] matching the input list
        agents: list[BaseSubAgent] = [
            SuccessAgent(domain_id="agent_a", delay=0.15),
            SuccessAgent(domain_id="agent_b", delay=0.05),
        ]

        results = await dispatcher.dispatch(
            agents=agents,
            query="Order test",
            context=empty_context,
            timeout_seconds=5.0,
        )

        assert results[0].domain_id == "agent_a"
        assert results[1].domain_id == "agent_b"
