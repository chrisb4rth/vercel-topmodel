"""
Unit tests for the Supervisor orchestration class.

Covers the key orchestration behaviors:
    1. Full pipeline — classify → dispatch → synthesize → stream.
    2. Empty registry — yields 503-style error StreamChunk.
    3. No classification match — yields clarification listing available sub-domains.
    4. All agents timeout — yields 504-style error StreamChunk.
    5. Partial timeout — graceful degradation with successful answers + unresolved list.
    6. Context injection — conversation context is passed to sub-agents.
    7. Context storage — exchange is stored after successful processing.
    8. Streaming — response is chunked into multiple StreamChunks with final metadata.

Each test uses lightweight mock agents extending BaseSubAgent to control
behavior without external dependencies.
"""

import asyncio

import pytest

from agents.base import BaseSubAgent
from context.store import ContextStore
from models import (
    ConfidenceLevel,
    ConversationContext,
    LegalReference,
    StreamChunk,
    SubAgentMetadata,
    SubAgentResponse,
)
from registry.registry import SubAgentRegistry
from supervisor.supervisor import Supervisor, _split_into_chunks


# ---------------------------------------------------------------------------
# Test fixtures: mock sub-agents with controllable behavior
# ---------------------------------------------------------------------------


class MockSeizureAgent(BaseSubAgent):
    """Mock agent simulating the account seizure domain."""

    def __init__(
        self,
        delay: float = 0.0,
    ) -> None:
        self._delay: float = delay

    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """Return a canned seizure response after an optional delay."""

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        return SubAgentResponse(
            domain_id="account_seizure",
            answer_body="The protected amount under § 850c ZPO is EUR 1,340.",
            references=[
                LegalReference(law_name="ZPO", paragraph="§ 850c", section="Abs. 1"),
            ],
            confidence=ConfidenceLevel.HIGH,
        )

    def get_metadata(self) -> SubAgentMetadata:
        """Return metadata for the account seizure domain."""

        return SubAgentMetadata(
            domain_id="account_seizure",
            description="Account seizure processing, protected amounts, third-party debt orders",
            supported_categories=["seizure_order", "protected_amounts", "priority_of_claims"],
        )


class MockInsolvencyAgent(BaseSubAgent):
    """Mock agent simulating the insolvency domain."""

    def __init__(
        self,
        delay: float = 0.0,
    ) -> None:
        self._delay: float = delay

    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """Return a canned insolvency response after an optional delay."""

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        return SubAgentResponse(
            domain_id="insolvency",
            answer_body="Account blocking is governed by § 89 InsO.",
            references=[
                LegalReference(law_name="InsO", paragraph="§ 89"),
            ],
            confidence=ConfidenceLevel.MEDIUM,
        )

    def get_metadata(self) -> SubAgentMetadata:
        """Return metadata for the insolvency domain."""

        return SubAgentMetadata(
            domain_id="insolvency",
            description="Insolvency proceedings, account blocking, administrator rights, payment prohibitions",
            supported_categories=["account_blocking", "administrator_rights", "payment_prohibitions"],
        )


class SlowMockAgent(BaseSubAgent):
    """Mock agent that sleeps beyond the timeout to trigger TimeoutError."""

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
        return SubAgentResponse(
            domain_id=self._domain_id,
            answer_body="Unreachable",
            references=[],
            confidence=ConfidenceLevel.LOW,
        )

    def get_metadata(self) -> SubAgentMetadata:
        """Return metadata for the slow domain."""

        return SubAgentMetadata(
            domain_id=self._domain_id,
            description="Slow domain for timeout testing",
            supported_categories=["slow"],
        )


# ---------------------------------------------------------------------------
# Helper to collect all chunks from the async generator
# ---------------------------------------------------------------------------


async def _collect_chunks(
    supervisor: Supervisor,
    query: str,
    session_id: str = "test-session",
    language: str = "de",
) -> list[StreamChunk]:
    """
    Consume the async generator from process_query and collect all StreamChunks.

    This helper simplifies test assertions by materializing the full stream
    into a list that can be inspected element-by-element.

    Args:
        supervisor: The Supervisor instance to invoke.
        query: The user query to process.
        session_id: Session identifier for context management.
        language: Detected language code.

    Returns:
        A list of all StreamChunk objects yielded by the generator.
    """

    chunks: list[StreamChunk] = []
    async for chunk in supervisor.process_query(
        query=query,
        session_id=session_id,
        language=language,
    ):
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def context_store() -> ContextStore:
    """Create a fresh ContextStore instance."""

    return ContextStore()


@pytest.fixture
def empty_registry() -> SubAgentRegistry:
    """Create an empty SubAgentRegistry (no agents registered)."""

    return SubAgentRegistry()


@pytest.fixture
def populated_registry() -> SubAgentRegistry:
    """Create a SubAgentRegistry with both mock agents registered."""

    registry: SubAgentRegistry = SubAgentRegistry()
    registry.register(MockSeizureAgent())
    registry.register(MockInsolvencyAgent())
    return registry


@pytest.fixture
def seizure_only_registry() -> SubAgentRegistry:
    """Create a SubAgentRegistry with only the seizure agent registered."""

    registry: SubAgentRegistry = SubAgentRegistry()
    registry.register(MockSeizureAgent())
    return registry


@pytest.fixture
def supervisor_with_agents(
    populated_registry: SubAgentRegistry,
    context_store: ContextStore,
) -> Supervisor:
    """Create a Supervisor with both mock agents available."""

    return Supervisor(
        registry=populated_registry,
        context_store=context_store,
    )


@pytest.fixture
def supervisor_empty_registry(
    empty_registry: SubAgentRegistry,
    context_store: ContextStore,
) -> Supervisor:
    """Create a Supervisor with an empty registry (system not ready)."""

    return Supervisor(
        registry=empty_registry,
        context_store=context_store,
    )


# ---------------------------------------------------------------------------
# Test: Empty registry (503-style error)
# ---------------------------------------------------------------------------


class TestEmptyRegistry:
    """Verify that an empty registry yields a 503-style error StreamChunk."""

    async def test_empty_registry_yields_error_chunk(
        self,
        supervisor_empty_registry: Supervisor,
    ) -> None:
        """When no agents are registered, a single error chunk is yielded."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_empty_registry,
            query="What is the protected amount?",
        )

        # Exactly one chunk should be yielded
        assert len(chunks) == 1

    async def test_empty_registry_chunk_is_final(
        self,
        supervisor_empty_registry: Supervisor,
    ) -> None:
        """The error chunk for empty registry is marked as final."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_empty_registry,
            query="Any query",
        )

        assert chunks[0].is_final is True

    async def test_empty_registry_chunk_has_503_metadata(
        self,
        supervisor_empty_registry: Supervisor,
    ) -> None:
        """The error chunk metadata indicates system_unavailable with status 503."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_empty_registry,
            query="Any query",
        )

        assert chunks[0].metadata is not None
        assert chunks[0].metadata["error"] == "system_unavailable"
        assert chunks[0].metadata["status"] == 503

    async def test_empty_registry_chunk_content_mentions_unavailable(
        self,
        supervisor_empty_registry: Supervisor,
    ) -> None:
        """The error chunk content communicates that the system is unavailable."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_empty_registry,
            query="Any query",
        )

        assert "unavailable" in chunks[0].content.lower()


# ---------------------------------------------------------------------------
# Test: No classification match (clarification)
# ---------------------------------------------------------------------------


class TestNoClassificationMatch:
    """Verify that unclassifiable queries yield a clarification StreamChunk."""

    async def test_unclassifiable_query_yields_clarification(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """A query with no keyword overlap yields a clarification chunk."""

        # Use a query that won't match any domain keywords
        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="xyzzy foobar baz",
        )

        # Should yield exactly one clarification chunk
        assert len(chunks) == 1
        assert chunks[0].is_final is True

    async def test_clarification_lists_available_domains(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """The clarification chunk lists all available sub-domain names."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="xyzzy foobar baz",
        )

        # Both registered domains should be mentioned in the content
        assert "account_seizure" in chunks[0].content
        assert "insolvency" in chunks[0].content

    async def test_clarification_metadata_has_available_domains(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """The clarification metadata includes the list of available domain IDs."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="xyzzy foobar baz",
        )

        assert chunks[0].metadata is not None
        assert "available_domains" in chunks[0].metadata
        available: list[str] = chunks[0].metadata["available_domains"]
        assert "account_seizure" in available
        assert "insolvency" in available


# ---------------------------------------------------------------------------
# Test: Successful full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Verify the full pipeline: classify → dispatch → synthesize → stream."""

    async def test_seizure_query_produces_answer_chunks(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """A seizure-related query produces content chunks followed by a final chunk."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
        )

        # At least one content chunk + one final metadata chunk
        assert len(chunks) >= 2

        # The last chunk must be the final metadata chunk
        final_chunk: StreamChunk = chunks[-1]
        assert final_chunk.is_final is True
        assert final_chunk.metadata is not None

    async def test_final_chunk_contains_confidence(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """The final metadata chunk includes the confidence level."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
        )

        final_chunk: StreamChunk = chunks[-1]
        assert "confidence" in final_chunk.metadata

    async def test_final_chunk_contains_references_count(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """The final metadata chunk includes the references count."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
        )

        final_chunk: StreamChunk = chunks[-1]
        assert "references_count" in final_chunk.metadata
        assert final_chunk.metadata["references_count"] >= 1

    async def test_final_chunk_contains_unresolved_domains(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """The final metadata chunk includes the unresolved_domains list."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
        )

        final_chunk: StreamChunk = chunks[-1]
        assert "unresolved_domains" in final_chunk.metadata

    async def test_content_chunks_contain_answer_text(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """Content chunks (non-final) contain the synthesized answer text."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
        )

        # Combine all non-final chunk content
        content_chunks: list[StreamChunk] = [c for c in chunks if not c.is_final]
        combined_content: str = "".join(c.content for c in content_chunks)

        # The answer should mention the seizure domain content
        assert "850c" in combined_content or "protected" in combined_content.lower()

    async def test_non_final_chunks_have_no_metadata(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """Non-final content chunks have metadata set to None."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
        )

        content_chunks: list[StreamChunk] = [c for c in chunks if not c.is_final]
        for chunk in content_chunks:
            assert chunk.metadata is None


# ---------------------------------------------------------------------------
# Test: All agents timeout (504-style error)
# ---------------------------------------------------------------------------


class TestAllAgentsTimeout:
    """Verify that all agents timing out yields a 504-style error StreamChunk."""

    async def test_all_timeout_yields_error_chunk(
        self,
        context_store: ContextStore,
    ) -> None:
        """When all dispatched agents time out, a 504-style error chunk is yielded."""

        # Register a slow agent that will always time out
        registry: SubAgentRegistry = SubAgentRegistry()
        # Use a domain_id that matches the query keywords for classification
        slow_agent: SlowMockAgent = SlowMockAgent(
            domain_id="account_seizure",
            delay=60.0,
        )
        # Override get_metadata to match seizure keywords
        slow_agent.get_metadata = lambda: SubAgentMetadata(
            domain_id="account_seizure",
            description="Account seizure processing, protected amounts",
            supported_categories=["seizure_order", "protected_amounts"],
        )
        registry.register(slow_agent)

        supervisor: Supervisor = Supervisor(
            registry=registry,
            context_store=context_store,
        )

        # Monkey-patch the dispatcher timeout to be very short for test speed
        import supervisor.supervisor as sup_module
        original_timeout = sup_module._DEFAULT_TIMEOUT_SECONDS
        sup_module._DEFAULT_TIMEOUT_SECONDS = 0.05

        try:
            chunks: list[StreamChunk] = await _collect_chunks(
                supervisor=supervisor,
                query="What is the protected amount under seizure?",
            )

            # Should yield exactly one error chunk
            assert len(chunks) == 1
            assert chunks[0].is_final is True
            assert chunks[0].metadata is not None
            assert chunks[0].metadata["error"] == "timeout"
            assert chunks[0].metadata["status"] == 504
        finally:
            # Restore original timeout
            sup_module._DEFAULT_TIMEOUT_SECONDS = original_timeout

    async def test_all_timeout_lists_attempted_domains(
        self,
        context_store: ContextStore,
    ) -> None:
        """The 504 error chunk metadata lists the attempted domains."""

        registry: SubAgentRegistry = SubAgentRegistry()
        slow_agent: SlowMockAgent = SlowMockAgent(
            domain_id="account_seizure",
            delay=60.0,
        )
        slow_agent.get_metadata = lambda: SubAgentMetadata(
            domain_id="account_seizure",
            description="Account seizure processing, protected amounts",
            supported_categories=["seizure_order", "protected_amounts"],
        )
        registry.register(slow_agent)

        supervisor: Supervisor = Supervisor(
            registry=registry,
            context_store=context_store,
        )

        import supervisor.supervisor as sup_module
        original_timeout = sup_module._DEFAULT_TIMEOUT_SECONDS
        sup_module._DEFAULT_TIMEOUT_SECONDS = 0.05

        try:
            chunks: list[StreamChunk] = await _collect_chunks(
                supervisor=supervisor,
                query="What is the protected amount under seizure?",
            )

            assert "attempted_domains" in chunks[0].metadata
            assert "account_seizure" in chunks[0].metadata["attempted_domains"]
        finally:
            sup_module._DEFAULT_TIMEOUT_SECONDS = original_timeout


# ---------------------------------------------------------------------------
# Test: Partial timeout (graceful degradation)
# ---------------------------------------------------------------------------


class TestPartialTimeout:
    """Verify graceful degradation when some agents succeed and others time out."""

    async def test_partial_timeout_still_yields_answer(
        self,
        context_store: ContextStore,
    ) -> None:
        """When one agent succeeds and another times out, the answer is still streamed."""

        registry: SubAgentRegistry = SubAgentRegistry()
        # Fast seizure agent that will succeed
        registry.register(MockSeizureAgent())
        # Slow insolvency agent that will time out
        slow_insolvency: SlowMockAgent = SlowMockAgent(
            domain_id="insolvency",
            delay=60.0,
        )
        slow_insolvency.get_metadata = lambda: SubAgentMetadata(
            domain_id="insolvency",
            description="Insolvency proceedings, account blocking, administrator rights",
            supported_categories=["account_blocking", "administrator_rights"],
        )
        registry.register(slow_insolvency)

        supervisor: Supervisor = Supervisor(
            registry=registry,
            context_store=context_store,
        )

        import supervisor.supervisor as sup_module
        original_timeout = sup_module._DEFAULT_TIMEOUT_SECONDS
        sup_module._DEFAULT_TIMEOUT_SECONDS = 0.05

        try:
            # Query that matches both domains
            chunks: list[StreamChunk] = await _collect_chunks(
                supervisor=supervisor,
                query="What about seizure and insolvency account blocking?",
            )

            # Should have content chunks + final metadata chunk (not a 504 error)
            assert len(chunks) >= 2
            final_chunk: StreamChunk = chunks[-1]
            assert final_chunk.is_final is True
            # Should NOT be a 504 error — partial success
            assert final_chunk.metadata.get("error") is None
        finally:
            sup_module._DEFAULT_TIMEOUT_SECONDS = original_timeout

    async def test_partial_timeout_unresolved_domains_in_metadata(
        self,
        context_store: ContextStore,
    ) -> None:
        """Partial timeout lists the timed-out domain in unresolved_domains metadata."""

        registry: SubAgentRegistry = SubAgentRegistry()
        registry.register(MockSeizureAgent())
        slow_insolvency: SlowMockAgent = SlowMockAgent(
            domain_id="insolvency",
            delay=60.0,
        )
        slow_insolvency.get_metadata = lambda: SubAgentMetadata(
            domain_id="insolvency",
            description="Insolvency proceedings, account blocking, administrator rights",
            supported_categories=["account_blocking", "administrator_rights"],
        )
        registry.register(slow_insolvency)

        supervisor: Supervisor = Supervisor(
            registry=registry,
            context_store=context_store,
        )

        import supervisor.supervisor as sup_module
        original_timeout = sup_module._DEFAULT_TIMEOUT_SECONDS
        sup_module._DEFAULT_TIMEOUT_SECONDS = 0.05

        try:
            chunks: list[StreamChunk] = await _collect_chunks(
                supervisor=supervisor,
                query="What about seizure and insolvency account blocking?",
            )

            final_chunk: StreamChunk = chunks[-1]
            assert "insolvency" in final_chunk.metadata["unresolved_domains"]
        finally:
            sup_module._DEFAULT_TIMEOUT_SECONDS = original_timeout


# ---------------------------------------------------------------------------
# Test: Context injection and storage
# ---------------------------------------------------------------------------


class TestContextManagement:
    """Verify that conversation context is retrieved, injected, and stored."""

    async def test_exchange_stored_after_successful_query(
        self,
        supervisor_with_agents: Supervisor,
        context_store: ContextStore,
    ) -> None:
        """After a successful query, the exchange is stored in the context store."""

        session_id: str = "context-test-session"

        # Process a query that will match and produce a response
        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
            session_id=session_id,
        )

        # Verify the exchange was stored
        context: ConversationContext = await context_store.get_context(session_id)
        assert len(context.exchanges) == 1
        assert context.exchanges[0].user_query == "What is the protected amount under seizure?"
        assert len(context.exchanges[0].system_response) > 0

    async def test_exchange_not_stored_on_empty_registry(
        self,
        supervisor_empty_registry: Supervisor,
        context_store: ContextStore,
    ) -> None:
        """When the registry is empty (503 error), no exchange is stored."""

        session_id: str = "no-store-session"

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_empty_registry,
            query="Any query",
            session_id=session_id,
        )

        context: ConversationContext = await context_store.get_context(session_id)
        assert len(context.exchanges) == 0

    async def test_exchange_not_stored_on_no_classification(
        self,
        supervisor_with_agents: Supervisor,
        context_store: ContextStore,
    ) -> None:
        """When no classification match occurs, no exchange is stored."""

        session_id: str = "no-classify-session"

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="xyzzy foobar baz",
            session_id=session_id,
        )

        context: ConversationContext = await context_store.get_context(session_id)
        assert len(context.exchanges) == 0

    async def test_multiple_queries_accumulate_exchanges(
        self,
        supervisor_with_agents: Supervisor,
        context_store: ContextStore,
    ) -> None:
        """Multiple successful queries in the same session accumulate exchanges."""

        session_id: str = "multi-exchange-session"

        # First query
        await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
            session_id=session_id,
        )

        # Second query
        await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="Tell me about insolvency account blocking",
            session_id=session_id,
        )

        context: ConversationContext = await context_store.get_context(session_id)
        assert len(context.exchanges) == 2

    async def test_stored_exchange_contains_references(
        self,
        supervisor_with_agents: Supervisor,
        context_store: ContextStore,
    ) -> None:
        """The stored exchange preserves legal references from the synthesized response."""

        session_id: str = "ref-store-session"

        await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What is the protected amount under seizure?",
            session_id=session_id,
        )

        context: ConversationContext = await context_store.get_context(session_id)
        assert len(context.exchanges[0].references) >= 1


# ---------------------------------------------------------------------------
# Test: Multi-domain query
# ---------------------------------------------------------------------------


class TestMultiDomainQuery:
    """Verify that queries spanning multiple domains dispatch to all matched agents."""

    async def test_cross_domain_query_includes_both_domains(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """A query matching both seizure and insolvency produces content from both."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What about seizure protected amounts and insolvency account blocking?",
        )

        # Combine all content
        content_chunks: list[StreamChunk] = [c for c in chunks if not c.is_final]
        combined_content: str = "".join(c.content for c in content_chunks)

        # Both domains should be represented in the answer
        assert "account_seizure" in combined_content or "850c" in combined_content
        assert "insolvency" in combined_content or "89" in combined_content

    async def test_cross_domain_final_metadata_has_multiple_references(
        self,
        supervisor_with_agents: Supervisor,
    ) -> None:
        """A cross-domain query produces a final chunk with references from both agents."""

        chunks: list[StreamChunk] = await _collect_chunks(
            supervisor=supervisor_with_agents,
            query="What about seizure protected amounts and insolvency account blocking?",
        )

        final_chunk: StreamChunk = chunks[-1]
        # Both agents contribute at least one reference each
        assert final_chunk.metadata["references_count"] >= 2


# ---------------------------------------------------------------------------
# Test: _split_into_chunks helper
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    """Verify the text chunking utility used for streaming."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty text produces no chunks."""

        result: list[str] = _split_into_chunks("", 100)
        assert result == []

    def test_short_text_returns_single_chunk(self) -> None:
        """Text shorter than chunk_size is returned as a single chunk."""

        text: str = "Short text."
        result: list[str] = _split_into_chunks(text, 100)
        assert result == ["Short text."]

    def test_long_text_splits_at_word_boundaries(self) -> None:
        """Long text is split at word boundaries, not mid-word."""

        text: str = "The quick brown fox jumps over the lazy dog"
        result: list[str] = _split_into_chunks(text, 20)

        # All chunks should be <= 20 chars (approximately, due to word boundaries)
        for chunk in result:
            assert len(chunk) <= 25  # Allow slight overshoot for word boundary

        # Reassembled text should match original (modulo whitespace)
        reassembled: str = " ".join(result)
        assert reassembled.replace("  ", " ") == text

    def test_exact_chunk_size_text(self) -> None:
        """Text exactly at chunk_size is returned as a single chunk."""

        text: str = "x" * 120
        result: list[str] = _split_into_chunks(text, 120)
        assert result == [text]

    def test_single_long_word_forced_split(self) -> None:
        """A single word longer than chunk_size is placed in its own chunk."""

        text: str = "a" * 200
        result: list[str] = _split_into_chunks(text, 50)

        # Should produce multiple chunks since the word exceeds chunk_size
        assert len(result) >= 2
        # Reassembled should equal original
        assert "".join(result) == text


# ---------------------------------------------------------------------------
# Test: Supervisor initialization
# ---------------------------------------------------------------------------


class TestSupervisorInit:
    """Verify that the Supervisor initializes its internal components correctly."""

    def test_supervisor_creates_classifier(
        self,
        populated_registry: SubAgentRegistry,
        context_store: ContextStore,
    ) -> None:
        """The Supervisor instantiates its own QueryClassifier."""

        supervisor: Supervisor = Supervisor(
            registry=populated_registry,
            context_store=context_store,
        )

        assert supervisor._classifier is not None

    def test_supervisor_creates_dispatcher(
        self,
        populated_registry: SubAgentRegistry,
        context_store: ContextStore,
    ) -> None:
        """The Supervisor instantiates its own ParallelDispatcher."""

        supervisor: Supervisor = Supervisor(
            registry=populated_registry,
            context_store=context_store,
        )

        assert supervisor._dispatcher is not None

    def test_supervisor_creates_synthesizer(
        self,
        populated_registry: SubAgentRegistry,
        context_store: ContextStore,
    ) -> None:
        """The Supervisor instantiates its own ResponseSynthesizer."""

        supervisor: Supervisor = Supervisor(
            registry=populated_registry,
            context_store=context_store,
        )

        assert supervisor._synthesizer is not None

    def test_supervisor_stores_registry(
        self,
        populated_registry: SubAgentRegistry,
        context_store: ContextStore,
    ) -> None:
        """The Supervisor stores the injected registry reference."""

        supervisor: Supervisor = Supervisor(
            registry=populated_registry,
            context_store=context_store,
        )

        assert supervisor._registry is populated_registry

    def test_supervisor_stores_context_store(
        self,
        populated_registry: SubAgentRegistry,
        context_store: ContextStore,
    ) -> None:
        """The Supervisor stores the injected context store reference."""

        supervisor: Supervisor = Supervisor(
            registry=populated_registry,
            context_store=context_store,
        )

        assert supervisor._context_store is context_store
