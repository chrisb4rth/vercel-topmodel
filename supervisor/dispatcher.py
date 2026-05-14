"""
Parallel dispatcher module for concurrent sub-agent invocation.

This module implements the ParallelDispatcher class, which is responsible for
dispatching a classified query to one or more sub-agents concurrently. Each
sub-agent is invoked with an independent per-agent timeout (default 30 seconds)
so that a slow or failing agent does not block the entire pipeline.

The dispatcher uses asyncio.gather with return_exceptions=True to run all
agent invocations in parallel, then wraps each outcome into a SubAgentResult:
    - Successful responses are stored directly.
    - TimeoutErrors produce a timed_out=True result.
    - Other exceptions produce an error result with the exception message.

This design ensures that the synthesizer always receives a complete list of
results — one per dispatched agent — enabling graceful degradation when only
some agents succeed.

Requirements: 2.1, 2.3, 2.6
"""

import asyncio
from typing import Any

from agents.base import BaseSubAgent
from models import ConversationContext, SubAgentResult


class ParallelDispatcher:
    """
    Dispatches queries to multiple sub-agents concurrently with per-agent timeouts.

    The dispatcher is stateless — it does not cache results or maintain references
    to agents between calls. Each invocation of `dispatch` creates fresh asyncio
    tasks, applies the timeout, and returns results in the same order as the input
    agent list. This positional correspondence allows the caller to correlate
    results back to the agents that produced them.

    Usage:
        dispatcher = ParallelDispatcher()
        results = await dispatcher.dispatch(
            agents=[seizure_agent, insolvency_agent],
            query="What is the protected amount?",
            context=conversation_context,
            timeout_seconds=30.0,
        )
    """

    async def dispatch(
        self,
        agents: list[BaseSubAgent],
        query: str,
        context: ConversationContext,
        timeout_seconds: float = 30.0,
    ) -> list[SubAgentResult]:
        """
        Invoke all agents in parallel with per-agent timeout enforcement.

        Creates one asyncio task per agent, wraps each with asyncio.wait_for
        to enforce the per-agent timeout, then gathers all tasks concurrently.
        Results are returned in the same positional order as the input agents
        list, regardless of completion order.

        Args:
            agents: List of sub-agents to invoke. Each must implement
                BaseSubAgent.handle_query and BaseSubAgent.get_metadata.
            query: The user's natural-language legal question to dispatch.
            context: Conversation history for the current session, passed
                to each agent so they can resolve follow-up references.
            timeout_seconds: Maximum seconds each individual agent is allowed
                to spend processing the query. Defaults to 30.0 per Req 2.3.

        Returns:
            A list of SubAgentResult instances, one per input agent, in the
            same order. Each result contains either a successful response,
            a timed_out flag, or an error message.
        """

        # Build a list of timeout-wrapped coroutines, one per agent
        wrapped_tasks: list[asyncio.Task[Any]] = [
            asyncio.wait_for(
                agent.handle_query(query, context),
                timeout=timeout_seconds,
            )
            for agent in agents
        ]

        # Execute all agent invocations concurrently; exceptions are returned
        # as values rather than raised, so we can inspect each outcome individually
        raw_outcomes: list[Any] = await asyncio.gather(
            *wrapped_tasks,
            return_exceptions=True,
        )

        # Convert each raw outcome into a uniform SubAgentResult
        results: list[SubAgentResult] = []
        for agent, outcome in zip(agents, raw_outcomes):
            # Retrieve the domain_id from agent metadata for result attribution
            domain_id: str = agent.get_metadata().domain_id

            result: SubAgentResult = _build_result_from_outcome(
                domain_id=domain_id,
                outcome=outcome,
            )
            results.append(result)

        return results


def _build_result_from_outcome(
    domain_id: str,
    outcome: Any,
) -> SubAgentResult:
    """
    Convert a raw gather outcome into a typed SubAgentResult.

    This helper centralizes the branching logic for the three possible
    outcome types: successful SubAgentResponse, asyncio.TimeoutError,
    or any other exception. Keeping this logic in a standalone function
    makes it easier to unit-test in isolation.

    Args:
        domain_id: The sub-domain identifier for the agent that produced
            this outcome, used to populate SubAgentResult.domain_id.
        outcome: The value returned by asyncio.gather for this agent's task.
            May be a SubAgentResponse (success), a TimeoutError, or another
            exception instance.

    Returns:
        A SubAgentResult reflecting the outcome type.
    """

    # Timeout: agent did not respond within the allowed window
    if isinstance(outcome, asyncio.TimeoutError):
        return SubAgentResult(
            domain_id=domain_id,
            timed_out=True,
        )

    # General exception: agent raised an error during processing
    if isinstance(outcome, BaseException):
        return SubAgentResult(
            domain_id=domain_id,
            error=str(outcome),
        )

    # Success: agent returned a valid SubAgentResponse
    return SubAgentResult(
        domain_id=domain_id,
        response=outcome,
    )
