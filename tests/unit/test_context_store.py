"""
Unit tests for the context store module.

Tests cover the ContextStore class and its three core operations:
- get_context: retrieves or creates empty conversation contexts
- append_exchange: adds exchanges to session history
- summarize_if_needed: triggers summarization when threshold is exceeded

Additional tests verify entity preservation (legal references, monetary amounts,
party names), the is_truncated flag, and round-trip data integrity.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import pytest

from context.store import (
    ContextStore,
    MAX_EXCHANGES_BEFORE_TRUNCATION,
    _extract_entities,
    _format_legal_reference,
)
from models import ConversationContext, Exchange, LegalReference


# ---------------------------------------------------------------------------
# get_context tests
# ---------------------------------------------------------------------------


class TestGetContext:
    """Tests for retrieving conversation context from the store."""

    @pytest.mark.asyncio
    async def test_returns_empty_context_for_new_session(self) -> None:
        """A session that has never been accessed returns an empty context."""

        store: ContextStore = ContextStore()

        context: ConversationContext = await store.get_context("new-session")

        assert context.exchanges == []
        assert context.summary is None
        assert context.is_truncated is False
        assert context.preserved_entities == []

    @pytest.mark.asyncio
    async def test_returns_same_context_on_repeated_access(self) -> None:
        """Accessing the same session_id twice returns the same object."""

        store: ContextStore = ContextStore()

        context_first: ConversationContext = await store.get_context("session-1")
        context_second: ConversationContext = await store.get_context("session-1")

        # Should be the exact same object in memory.
        assert context_first is context_second

    @pytest.mark.asyncio
    async def test_different_sessions_are_isolated(self) -> None:
        """Different session_ids produce independent contexts."""

        store: ContextStore = ContextStore()

        context_a: ConversationContext = await store.get_context("session-a")
        context_b: ConversationContext = await store.get_context("session-b")

        # Mutating one should not affect the other.
        context_a.exchanges.append(
            Exchange(user_query="test", system_response="response")
        )

        assert len(context_a.exchanges) == 1
        assert len(context_b.exchanges) == 0


# ---------------------------------------------------------------------------
# append_exchange tests
# ---------------------------------------------------------------------------


class TestAppendExchange:
    """Tests for appending exchanges to session history."""

    @pytest.mark.asyncio
    async def test_appends_single_exchange(self) -> None:
        """A single exchange is stored and retrievable."""

        store: ContextStore = ContextStore()
        exchange: Exchange = Exchange(
            user_query="Was ist der Pfändungsfreibetrag?",
            system_response="Der Pfändungsfreibetrag beträgt...",
            references=[
                LegalReference(law_name="ZPO", paragraph="§ 850c")
            ],
        )

        await store.append_exchange("session-1", exchange)
        context: ConversationContext = await store.get_context("session-1")

        assert len(context.exchanges) == 1
        assert context.exchanges[0] is exchange

    @pytest.mark.asyncio
    async def test_appends_multiple_exchanges_in_order(self) -> None:
        """Multiple exchanges are stored in insertion order."""

        store: ContextStore = ContextStore()
        exchanges: list[Exchange] = [
            Exchange(user_query=f"Query {i}", system_response=f"Response {i}")
            for i in range(5)
        ]

        for exchange in exchanges:
            await store.append_exchange("session-1", exchange)

        context: ConversationContext = await store.get_context("session-1")

        assert len(context.exchanges) == 5
        for i, stored_exchange in enumerate(context.exchanges):
            assert stored_exchange.user_query == f"Query {i}"
            assert stored_exchange.system_response == f"Response {i}"

    @pytest.mark.asyncio
    async def test_creates_session_if_not_exists(self) -> None:
        """Appending to a non-existent session creates it implicitly."""

        store: ContextStore = ContextStore()
        exchange: Exchange = Exchange(
            user_query="First question",
            system_response="First answer",
        )

        await store.append_exchange("brand-new-session", exchange)
        context: ConversationContext = await store.get_context("brand-new-session")

        assert len(context.exchanges) == 1
        assert context.exchanges[0].user_query == "First question"


# ---------------------------------------------------------------------------
# Round-trip preservation tests
# ---------------------------------------------------------------------------


class TestRoundTripPreservation:
    """Tests verifying that exchange data survives storage and retrieval intact."""

    @pytest.mark.asyncio
    async def test_preserves_user_query_text(self) -> None:
        """The user_query field is stored and retrieved without modification."""

        store: ContextStore = ContextStore()
        original_query: str = "Wie hoch ist der Pfändungsfreibetrag nach § 850c ZPO?"
        exchange: Exchange = Exchange(
            user_query=original_query,
            system_response="Der Grundfreibetrag...",
        )

        await store.append_exchange("session-rt", exchange)
        context: ConversationContext = await store.get_context("session-rt")

        assert context.exchanges[0].user_query == original_query

    @pytest.mark.asyncio
    async def test_preserves_system_response_text(self) -> None:
        """The system_response field is stored and retrieved without modification."""

        store: ContextStore = ContextStore()
        original_response: str = (
            "Gemäß § 850c ZPO beträgt der Grundfreibetrag EUR 1.402,28 monatlich."
        )
        exchange: Exchange = Exchange(
            user_query="Freibetrag?",
            system_response=original_response,
        )

        await store.append_exchange("session-rt", exchange)
        context: ConversationContext = await store.get_context("session-rt")

        assert context.exchanges[0].system_response == original_response

    @pytest.mark.asyncio
    async def test_preserves_legal_references(self) -> None:
        """Legal references attached to exchanges survive round-trip."""

        store: ContextStore = ContextStore()
        refs: list[LegalReference] = [
            LegalReference(law_name="ZPO", paragraph="§ 850c", section="Abs. 1"),
            LegalReference(law_name="InsO", paragraph="§ 80"),
        ]
        exchange: Exchange = Exchange(
            user_query="Question",
            system_response="Answer",
            references=refs,
        )

        await store.append_exchange("session-rt", exchange)
        context: ConversationContext = await store.get_context("session-rt")

        assert len(context.exchanges[0].references) == 2
        assert context.exchanges[0].references[0].law_name == "ZPO"
        assert context.exchanges[0].references[0].paragraph == "§ 850c"
        assert context.exchanges[0].references[0].section == "Abs. 1"
        assert context.exchanges[0].references[1].law_name == "InsO"
        assert context.exchanges[0].references[1].paragraph == "§ 80"
        assert context.exchanges[0].references[1].section is None


# ---------------------------------------------------------------------------
# summarize_if_needed tests
# ---------------------------------------------------------------------------


class TestSummarizeIfNeeded:
    """Tests for context summarization behavior."""

    @pytest.mark.asyncio
    async def test_no_summarization_below_threshold(self) -> None:
        """Sessions with exchanges at or below the threshold are not summarized."""

        store: ContextStore = ContextStore()

        # Add exactly MAX_EXCHANGES_BEFORE_TRUNCATION exchanges (the threshold).
        for i in range(MAX_EXCHANGES_BEFORE_TRUNCATION):
            exchange: Exchange = Exchange(
                user_query=f"Query {i}",
                system_response=f"Response {i}",
            )
            await store.append_exchange("session-no-trunc", exchange)

        result: bool = await store.summarize_if_needed("session-no-trunc")

        assert result is False
        context: ConversationContext = await store.get_context("session-no-trunc")
        assert context.is_truncated is False
        assert context.summary is None
        assert len(context.exchanges) == MAX_EXCHANGES_BEFORE_TRUNCATION

    @pytest.mark.asyncio
    async def test_summarization_triggers_at_threshold_plus_one(self) -> None:
        """Summarization triggers when exchanges exceed the threshold (21)."""

        store: ContextStore = ContextStore()

        # Add one more than the threshold to trigger summarization.
        for i in range(MAX_EXCHANGES_BEFORE_TRUNCATION + 1):
            exchange: Exchange = Exchange(
                user_query=f"Query {i}",
                system_response=f"Response {i}",
            )
            await store.append_exchange("session-trunc", exchange)

        result: bool = await store.summarize_if_needed("session-trunc")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_truncated_flag_set_after_summarization(self) -> None:
        """The is_truncated flag is True after summarization occurs."""

        store: ContextStore = ContextStore()

        for i in range(MAX_EXCHANGES_BEFORE_TRUNCATION + 1):
            exchange: Exchange = Exchange(
                user_query=f"Query {i}",
                system_response=f"Response {i}",
            )
            await store.append_exchange("session-flag", exchange)

        await store.summarize_if_needed("session-flag")
        context: ConversationContext = await store.get_context("session-flag")

        assert context.is_truncated is True

    @pytest.mark.asyncio
    async def test_summary_is_populated_after_summarization(self) -> None:
        """The summary field contains text after summarization."""

        store: ContextStore = ContextStore()

        for i in range(MAX_EXCHANGES_BEFORE_TRUNCATION + 1):
            exchange: Exchange = Exchange(
                user_query=f"Query {i}",
                system_response=f"Response {i}",
            )
            await store.append_exchange("session-summary", exchange)

        await store.summarize_if_needed("session-summary")
        context: ConversationContext = await store.get_context("session-summary")

        assert context.summary is not None
        assert len(context.summary) > 0

    @pytest.mark.asyncio
    async def test_recent_exchanges_retained_after_summarization(self) -> None:
        """After summarization, only the most recent exchanges are retained."""

        store: ContextStore = ContextStore()

        for i in range(MAX_EXCHANGES_BEFORE_TRUNCATION + 1):
            exchange: Exchange = Exchange(
                user_query=f"Query {i}",
                system_response=f"Response {i}",
            )
            await store.append_exchange("session-retain", exchange)

        await store.summarize_if_needed("session-retain")
        context: ConversationContext = await store.get_context("session-retain")

        # Should retain the last 10 exchanges (EXCHANGES_TO_RETAIN).
        assert len(context.exchanges) == 10
        # The retained exchanges should be the most recent ones.
        assert context.exchanges[0].user_query == "Query 11"
        assert context.exchanges[-1].user_query == "Query 20"

    @pytest.mark.asyncio
    async def test_returns_false_for_empty_session(self) -> None:
        """An empty session does not trigger summarization."""

        store: ContextStore = ContextStore()

        result: bool = await store.summarize_if_needed("empty-session")

        assert result is False


# ---------------------------------------------------------------------------
# Entity preservation tests
# ---------------------------------------------------------------------------


class TestEntityPreservation:
    """Tests verifying that key entities are preserved during summarization."""

    @pytest.mark.asyncio
    async def test_preserves_legal_references(self) -> None:
        """Legal references from summarized exchanges appear in preserved_entities."""

        store: ContextStore = ContextStore()

        # Add exchanges with legal references in the older portion.
        for i in range(MAX_EXCHANGES_BEFORE_TRUNCATION + 1):
            refs: list[LegalReference] = []
            if i < 5:
                # Only the first 5 exchanges have references (they'll be summarized).
                refs = [LegalReference(law_name="ZPO", paragraph=f"§ {850 + i}")]
            exchange: Exchange = Exchange(
                user_query=f"Query {i}",
                system_response=f"Response {i}",
                references=refs,
            )
            await store.append_exchange("session-entities", exchange)

        await store.summarize_if_needed("session-entities")
        context: ConversationContext = await store.get_context("session-entities")

        # The legal references from the first 5 exchanges should be preserved.
        assert len(context.preserved_entities) >= 5
        # Check that at least one formatted reference is present.
        assert any("ZPO" in entity for entity in context.preserved_entities)

    @pytest.mark.asyncio
    async def test_preserves_monetary_amounts(self) -> None:
        """Monetary amounts (EUR/€) from summarized exchanges are preserved."""

        store: ContextStore = ContextStore()

        # Add exchanges with monetary amounts in the older portion.
        for i in range(MAX_EXCHANGES_BEFORE_TRUNCATION + 1):
            response: str = f"Response {i}"
            if i == 0:
                response = "Der Freibetrag beträgt EUR 1.402,28 monatlich."
            if i == 1:
                response = "Die Forderung beläuft sich auf €5.000."
            exchange: Exchange = Exchange(
                user_query=f"Query {i}",
                system_response=response,
            )
            await store.append_exchange("session-money", exchange)

        await store.summarize_if_needed("session-money")
        context: ConversationContext = await store.get_context("session-money")

        # Monetary amounts should be in preserved entities.
        entities_text: str = " ".join(context.preserved_entities)
        assert "EUR" in entities_text or "€" in entities_text

    @pytest.mark.asyncio
    async def test_preserves_party_names(self) -> None:
        """Capitalized multi-word names from summarized exchanges are preserved."""

        store: ContextStore = ContextStore()

        # Add exchanges with party names in the older portion.
        for i in range(MAX_EXCHANGES_BEFORE_TRUNCATION + 1):
            query: str = f"Query {i}"
            if i == 0:
                query = "Was passiert mit dem Konto von Max Mustermann?"
            if i == 1:
                query = "Deutsche Bank AG hat eine Pfändung erhalten."
            exchange: Exchange = Exchange(
                user_query=query,
                system_response=f"Response {i}",
            )
            await store.append_exchange("session-names", exchange)

        await store.summarize_if_needed("session-names")
        context: ConversationContext = await store.get_context("session-names")

        # Party names should be in preserved entities.
        assert any(
            "Max Mustermann" in entity for entity in context.preserved_entities
        )
        assert any(
            "Deutsche Bank" in entity for entity in context.preserved_entities
        )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestFormatLegalReference:
    """Tests for the _format_legal_reference helper."""

    def test_formats_reference_without_section(self) -> None:
        """Reference without section produces 'paragraph law_name' format."""

        ref: LegalReference = LegalReference(law_name="ZPO", paragraph="§ 850c")

        result: str = _format_legal_reference(ref)

        assert result == "§ 850c ZPO"

    def test_formats_reference_with_section(self) -> None:
        """Reference with section includes it between paragraph and law_name."""

        ref: LegalReference = LegalReference(
            law_name="InsO", paragraph="§ 80", section="Abs. 1"
        )

        result: str = _format_legal_reference(ref)

        assert result == "§ 80 Abs. 1 InsO"


class TestExtractEntities:
    """Tests for the _extract_entities helper."""

    def test_extracts_legal_references(self) -> None:
        """Legal references from Exchange.references are extracted."""

        exchanges: list[Exchange] = [
            Exchange(
                user_query="Question",
                system_response="Answer",
                references=[
                    LegalReference(law_name="ZPO", paragraph="§ 850c"),
                ],
            )
        ]

        entities: list[str] = _extract_entities(exchanges)

        assert "§ 850c ZPO" in entities

    def test_extracts_monetary_amounts(self) -> None:
        """EUR/€ amounts in text are extracted."""

        exchanges: list[Exchange] = [
            Exchange(
                user_query="Wie hoch ist EUR 1.000?",
                system_response="Der Betrag ist €500.",
            )
        ]

        entities: list[str] = _extract_entities(exchanges)

        assert any("EUR" in e or "€" in e for e in entities)

    def test_deduplicates_entities(self) -> None:
        """Repeated entities appear only once in the result."""

        exchanges: list[Exchange] = [
            Exchange(
                user_query="EUR 500 question",
                system_response="EUR 500 answer",
            ),
            Exchange(
                user_query="Another EUR 500 question",
                system_response="Response",
            ),
        ]

        entities: list[str] = _extract_entities(exchanges)

        # EUR 500 should appear only once despite multiple occurrences.
        eur_entities: list[str] = [e for e in entities if "EUR 500" in e]
        assert len(eur_entities) == 1
