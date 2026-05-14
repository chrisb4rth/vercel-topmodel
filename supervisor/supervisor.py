"""
Supervisor orchestration module.

Provides the Supervisor class responsible for orchestrating the full query
processing pipeline: classification, parallel dispatch to sub-agents, response
synthesis, context management, and streaming delivery. The supervisor is the
central coordination point that wires together all system components without
containing domain-specific logic itself.

The processing pipeline for each query:
    1. Retrieve conversation context for the session from the context store.
    2. Discover available sub-agents via the registry's metadata.
    3. Reject the query if the registry is empty (system not ready — 503).
    4. Classify the query into one or more legal sub-domains.
    5. If no domains matched, yield a clarification listing available sub-domains.
    6. Retrieve the matched sub-agents from the registry.
    7. Dispatch the query to matched agents in parallel with context injection.
    8. Synthesize the dispatch results into a unified response.
    9. Store the exchange (query + answer) in the context store for follow-ups.
    10. Stream the synthesized response as chunked StreamChunks.
    11. Yield a final metadata chunk with confidence, reference count, and
        unresolved domains.

Edge cases handled:
    - Empty registry → error StreamChunk indicating system unavailable (503-style).
    - All agents time out → error StreamChunk indicating total timeout (504-style).
    - Partial timeouts → graceful degradation with successful answers + unresolved list.
    - No classification match → clarification StreamChunk listing available sub-domains.

Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.4, 2.5, 8.3
"""

import logging
from collections.abc import AsyncIterator

from context.store import ContextStore
from models import (
    ClassificationResult,
    ConversationContext,
    Exchange,
    StreamChunk,
    SubAgentMetadata,
    SubAgentResult,
    SynthesizedResponse,
)
from registry.registry import SubAgentRegistry
from supervisor.classifier import QueryClassifier
from supervisor.dispatcher import ParallelDispatcher
from supervisor.synthesizer import ResponseSynthesizer

# Module-level logger for supervisor orchestration diagnostics.
logger: logging.Logger = logging.getLogger(__name__)

# Default per-agent timeout in seconds (Requirement 2.3).
_DEFAULT_TIMEOUT_SECONDS: float = 30.0

# Maximum number of characters per stream chunk when simulating token streaming.
# Chosen to approximate sentence-level granularity for readable incremental delivery.
_STREAM_CHUNK_SIZE: int = 120


class Supervisor:
    """
    Orchestrates query classification, sub-agent delegation, and response synthesis.

    The Supervisor is the central coordination layer that connects:
        - SubAgentRegistry: discovers available sub-agents and their metadata.
        - QueryClassifier: determines which sub-domains a query belongs to.
        - ParallelDispatcher: invokes matched sub-agents concurrently with timeouts.
        - ResponseSynthesizer: merges partial responses into a coherent answer.
        - ContextStore: manages per-session conversation history for follow-ups.

    The supervisor does NOT contain domain-specific legal logic — it delegates
    all legal reasoning to sub-agents. Its role is purely orchestrational:
    routing, timing, error handling, and streaming.

    The classifier, dispatcher, and synthesizer are instantiated internally
    because they are stateless utilities that do not require external configuration.
    The registry and context store are injected because they carry state that
    must be shared across the application lifecycle.

    Attributes:
        _registry: The sub-agent registry providing agent discovery and lookup.
        _context_store: The per-session conversation history manager.
        _classifier: Internal query classifier for domain routing.
        _dispatcher: Internal parallel dispatcher for concurrent agent invocation.
        _synthesizer: Internal response synthesizer for merging sub-agent outputs.
    """

    def __init__(
        self,
        registry: SubAgentRegistry,
        context_store: ContextStore,
    ) -> None:
        """
        Initialize the Supervisor with its required dependencies.

        The registry and context store are injected because they carry shared
        application state. The classifier, dispatcher, and synthesizer are
        created internally as stateless utilities.

        Args:
            registry: The sub-agent registry used to discover available agents
                and retrieve agents for classified domains.
            context_store: The session context manager used to retrieve and
                store conversation history for follow-up queries.
        """

        # Injected dependencies carrying shared application state.
        self._registry: SubAgentRegistry = registry
        self._context_store: ContextStore = context_store

        # Internal stateless utilities instantiated by the supervisor.
        self._classifier: QueryClassifier = QueryClassifier()
        self._dispatcher: ParallelDispatcher = ParallelDispatcher()
        self._synthesizer: ResponseSynthesizer = ResponseSynthesizer()

        return None

    async def process_query(
        self,
        query: str,
        session_id: str,
        language: str,
    ) -> AsyncIterator[StreamChunk]:
        """
        Orchestrate the full query processing pipeline and stream the response.

        This is the primary entry point for processing a validated user query.
        It coordinates classification, dispatch, synthesis, context storage,
        and streaming delivery. The method is an async generator that yields
        StreamChunk instances for incremental delivery to the client.

        The pipeline handles several edge cases:
            - Empty registry: yields an error chunk (503-style) and returns.
            - No classification match: yields a clarification chunk listing
              available sub-domains and returns.
            - All agents timeout: yields an error chunk (504-style) and returns.
            - Partial timeouts: yields the successful portion with metadata
              indicating which domains were unresolved.

        Args:
            query: The user's natural-language legal question, already validated
                for length and language by the executor layer.
            session_id: Unique identifier for the conversation session, used
                to retrieve and store conversation context.
            language: The detected language code (e.g., "de", "en") for the query.

        Yields:
            StreamChunk instances containing response content fragments and a
            final chunk with metadata (confidence, references count, unresolved).
        """

        logger.info(
            "Processing query for session '%s' (language=%s, length=%d).",
            session_id,
            language,
            len(query),
        )

        # Step 1: Retrieve conversation context for follow-up resolution.
        context: ConversationContext = await self._context_store.get_context(session_id)

        # Step 2: Discover all available sub-agents from the registry.
        available_domains: list[SubAgentMetadata] = self._registry.get_all_metadata()

        # Step 3: If registry is empty, the system is not ready (Req 8.3).
        if not available_domains:
            logger.error(
                "Registry is empty — cannot process query. "
                "Rejecting with 503-style error for session '%s'.",
                session_id,
            )
            error_chunk: StreamChunk = StreamChunk(
                content="System unavailable: no sub-agents are registered. "
                "The system cannot process queries at this time.",
                is_final=True,
                metadata={"error": "system_unavailable", "status": 503},
            )
            yield error_chunk
            return

        # Step 4: Classify the query into one or more legal sub-domains.
        classification: ClassificationResult = await self._classifier.classify_query(
            query=query,
            available_domains=available_domains,
        )

        logger.info(
            "Classification result: domains=%s, confidence=%.2f.",
            classification.domain_ids,
            classification.confidence,
        )

        # Step 5: If no domains matched, yield a clarification listing available sub-domains.
        if not classification.domain_ids:
            available_names: list[str] = [
                f"{meta.domain_id} ({meta.description})"
                for meta in available_domains
            ]
            clarification_text: str = (
                "I could not determine which legal sub-domain your question belongs to. "
                "The following sub-domains are available:\n"
                + "\n".join(f"  - {name}" for name in available_names)
                + "\n\nPlease rephrase your question or specify the relevant domain."
            )
            clarification_chunk: StreamChunk = StreamChunk(
                content=clarification_text,
                is_final=True,
                metadata={"error": "no_classification_match", "available_domains": [m.domain_id for m in available_domains]},
            )
            yield clarification_chunk
            return

        # Step 6: Retrieve the matched sub-agents from the registry.
        from agents.base import BaseSubAgent
        matched_agents: list[BaseSubAgent] = self._registry.get_agents_for_domains(
            domain_ids=classification.domain_ids,
        )

        # Step 7: Dispatch to matched agents in parallel with context injection.
        dispatch_results: list[SubAgentResult] = await self._dispatcher.dispatch(
            agents=matched_agents,
            query=query,
            context=context,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

        # Step 8: Synthesize the dispatch results into a unified response.
        synthesized: SynthesizedResponse = await self._synthesizer.synthesize(
            results=dispatch_results,
            query=query,
        )

        # Check if ALL agents failed (total timeout / error — 504-style).
        all_failed: bool = (
            not synthesized.answer_body.strip()
            and len(synthesized.unresolved_domains) == len(matched_agents)
            and len(matched_agents) > 0
        )

        if all_failed:
            logger.error(
                "All dispatched sub-agents failed for session '%s'. "
                "Returning 504-style error. Attempted domains: %s.",
                session_id,
                synthesized.unresolved_domains,
            )
            timeout_chunk: StreamChunk = StreamChunk(
                content="No sub-domain could be resolved. All dispatched agents "
                "failed to respond within the allowed time.",
                is_final=True,
                metadata={
                    "error": "timeout",
                    "status": 504,
                    "attempted_domains": synthesized.unresolved_domains,
                },
            )
            yield timeout_chunk
            return

        # Step 9: Store the exchange in context for future follow-up queries.
        exchange: Exchange = Exchange(
            user_query=query,
            system_response=synthesized.answer_body,
            references=synthesized.references,
        )
        await self._context_store.append_exchange(
            session_id=session_id,
            exchange=exchange,
        )

        # Trigger summarization if the context window limit is reached.
        await self._context_store.summarize_if_needed(session_id=session_id)

        # Step 10: Stream the synthesized response as chunked StreamChunks.
        answer_text: str = synthesized.answer_body
        chunks: list[str] = _split_into_chunks(answer_text, _STREAM_CHUNK_SIZE)

        for chunk_text in chunks:
            content_chunk: StreamChunk = StreamChunk(
                content=chunk_text,
                is_final=False,
                metadata=None,
            )
            yield content_chunk

        # Step 11: Yield a final metadata chunk with response quality indicators.
        final_metadata: dict = {
            "confidence": synthesized.confidence.value,
            "references_count": len(synthesized.references),
            "unresolved_domains": synthesized.unresolved_domains,
            "recommend_professional": synthesized.recommend_professional,
        }
        final_chunk: StreamChunk = StreamChunk(
            content="",
            is_final=True,
            metadata=final_metadata,
        )
        yield final_chunk

        logger.info(
            "Completed query processing for session '%s'. "
            "Confidence=%s, references=%d, unresolved=%s.",
            session_id,
            synthesized.confidence.value,
            len(synthesized.references),
            synthesized.unresolved_domains,
        )

    async def classify_query(
        self,
        query: str,
        available_domains: list[SubAgentMetadata],
    ) -> ClassificationResult:
        """
        Classify a query into one or more legal sub-domains.

        Delegates to the internal QueryClassifier instance. This method is
        exposed on the Supervisor interface to allow direct classification
        calls from tests or other components that need classification without
        the full pipeline.

        Args:
            query: The user's natural-language legal question.
            available_domains: Metadata for all registered sub-agents, used
                by the classifier to determine domain matches.

        Returns:
            A ClassificationResult containing matched domain_ids, confidence,
            and detected language.
        """

        classification_result: ClassificationResult = await self._classifier.classify_query(
            query=query,
            available_domains=available_domains,
        )
        return classification_result

    async def synthesize_responses(
        self,
        partial_responses: list[SubAgentResult],
        query: str,
    ) -> SynthesizedResponse:
        """
        Synthesize multiple sub-agent dispatch results into a single response.

        Delegates to the internal ResponseSynthesizer instance. This method is
        exposed on the Supervisor interface to allow direct synthesis calls from
        tests or other components that need synthesis without the full pipeline.

        Args:
            partial_responses: List of SubAgentResult instances from the dispatcher,
                each containing either a successful response, timeout flag, or error.
            query: The original user query, retained for synthesis context.

        Returns:
            A SynthesizedResponse containing the merged answer, references,
            confidence, unresolved domains, and professional consultation flag.
        """

        synthesized_response: SynthesizedResponse = await self._synthesizer.synthesize(
            results=partial_responses,
            query=query,
        )
        return synthesized_response


def _split_into_chunks(
    text: str,
    chunk_size: int,
) -> list[str]:
    """
    Split a text string into chunks of approximately chunk_size characters.

    Attempts to split at word boundaries to avoid breaking words mid-stream.
    If a single word exceeds chunk_size, it is placed in its own chunk to
    avoid infinite loops.

    Args:
        text: The full text to split into streaming chunks.
        chunk_size: Target maximum characters per chunk. Actual chunks may be
            slightly shorter due to word-boundary splitting.

    Returns:
        A list of text chunks suitable for incremental streaming delivery.
        Returns a single-element list with the full text if it fits in one chunk.
        Returns an empty list if the input text is empty.
    """

    # Handle empty text edge case.
    if not text:
        return []

    # If the text fits in a single chunk, return it directly.
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    remaining: str = text

    while remaining:
        # If the remaining text fits in one chunk, take it all.
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break

        # Find the last space within the chunk_size window for word-boundary split.
        split_point: int = remaining.rfind(" ", 0, chunk_size)

        # If no space found (single long word), force split at chunk_size.
        if split_point == -1:
            split_point = chunk_size

        # Extract the chunk and advance the remaining text.
        chunk: str = remaining[:split_point]
        chunks.append(chunk)

        # Skip the space character at the split point when advancing.
        remaining = remaining[split_point:].lstrip(" ")

    return chunks
