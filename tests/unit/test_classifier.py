"""
Unit tests for the QueryClassifier.

Validates core classification behaviors:
    - Matching queries to domains via keyword overlap.
    - Returning empty domain_ids when no domain matches.
    - Cross-domain classification for multi-domain queries.
    - Confidence computation based on hit ratio.
    - Language detection via German stopword heuristics.
    - Guarantee that returned domain_ids exist in available_domains (Property 3).

Requirements: 1.1, 1.2, 1.3
"""

import pytest

from models import ClassificationResult, Language, SubAgentMetadata
from supervisor.classifier import QueryClassifier


@pytest.fixture
def classifier() -> QueryClassifier:
    """Provide a fresh QueryClassifier instance for each test."""

    return QueryClassifier()


@pytest.fixture
def account_seizure_metadata() -> SubAgentMetadata:
    """Metadata for the account seizure sub-agent."""

    return SubAgentMetadata(
        domain_id="account_seizure",
        description="Handles questions about account seizures, Kontopfändungen, seizure orders, and protected amounts",
        supported_categories=["seizure_order", "protected_amounts", "third_party_debt", "priority_of_claims"],
    )


@pytest.fixture
def insolvency_metadata() -> SubAgentMetadata:
    """Metadata for the insolvency sub-agent."""

    return SubAgentMetadata(
        domain_id="insolvency",
        description="Handles questions about insolvency proceedings, Insolvenzverfahren, account blocking, and administrator rights",
        supported_categories=["account_blocking", "administrator_rights", "payment_prohibitions", "estate_segregation"],
    )


@pytest.mark.asyncio
async def test_classify_single_domain_match(
    classifier: QueryClassifier,
    account_seizure_metadata: SubAgentMetadata,
    insolvency_metadata: SubAgentMetadata,
) -> None:
    """A query about seizure should match the account_seizure domain."""

    available: list[SubAgentMetadata] = [account_seizure_metadata, insolvency_metadata]

    result: ClassificationResult = await classifier.classify_query(
        query="What are the protected amounts for account seizure?",
        available_domains=available,
    )

    assert "account_seizure" in result.domain_ids
    assert result.confidence > 0.0

    return None


@pytest.mark.asyncio
async def test_classify_no_match_returns_empty(
    classifier: QueryClassifier,
    account_seizure_metadata: SubAgentMetadata,
    insolvency_metadata: SubAgentMetadata,
) -> None:
    """A query unrelated to any domain should return empty domain_ids."""

    available: list[SubAgentMetadata] = [account_seizure_metadata, insolvency_metadata]

    result: ClassificationResult = await classifier.classify_query(
        query="What is the weather today?",
        available_domains=available,
    )

    assert result.domain_ids == []
    assert result.confidence == 0.0

    return None


@pytest.mark.asyncio
async def test_classify_cross_domain_query(
    classifier: QueryClassifier,
    account_seizure_metadata: SubAgentMetadata,
    insolvency_metadata: SubAgentMetadata,
) -> None:
    """A query spanning both domains should return both domain_ids."""

    available: list[SubAgentMetadata] = [account_seizure_metadata, insolvency_metadata]

    # This query mentions both seizure and insolvency keywords.
    result: ClassificationResult = await classifier.classify_query(
        query="How does account seizure interact with insolvency proceedings?",
        available_domains=available,
    )

    assert "account_seizure" in result.domain_ids
    assert "insolvency" in result.domain_ids

    return None


@pytest.mark.asyncio
async def test_classify_german_language_detection(
    classifier: QueryClassifier,
    account_seizure_metadata: SubAgentMetadata,
) -> None:
    """A German query should be detected as Language.GERMAN."""

    available: list[SubAgentMetadata] = [account_seizure_metadata]

    result: ClassificationResult = await classifier.classify_query(
        query="Was sind die Pfändungsfreigrenzen für eine Kontopfändung?",
        available_domains=available,
    )

    assert result.language == Language.GERMAN

    return None


@pytest.mark.asyncio
async def test_classify_english_language_detection(
    classifier: QueryClassifier,
    account_seizure_metadata: SubAgentMetadata,
) -> None:
    """An English query should be detected as Language.ENGLISH."""

    available: list[SubAgentMetadata] = [account_seizure_metadata]

    result: ClassificationResult = await classifier.classify_query(
        query="What are the protected amounts for seizure?",
        available_domains=available,
    )

    assert result.language == Language.ENGLISH

    return None


@pytest.mark.asyncio
async def test_classify_empty_available_domains(
    classifier: QueryClassifier,
) -> None:
    """When no domains are available, return empty classification."""

    result: ClassificationResult = await classifier.classify_query(
        query="Tell me about account seizure",
        available_domains=[],
    )

    assert result.domain_ids == []
    assert result.confidence == 0.0

    return None


@pytest.mark.asyncio
async def test_all_returned_domain_ids_exist_in_available(
    classifier: QueryClassifier,
    account_seizure_metadata: SubAgentMetadata,
    insolvency_metadata: SubAgentMetadata,
) -> None:
    """Property 3: Every returned domain_id must exist in available_domains."""

    available: list[SubAgentMetadata] = [account_seizure_metadata, insolvency_metadata]
    valid_ids: set[str] = {meta.domain_id for meta in available}

    result: ClassificationResult = await classifier.classify_query(
        query="seizure insolvency blocking order",
        available_domains=available,
    )

    # Every returned domain_id must be in the available set.
    for domain_id in result.domain_ids:
        assert domain_id in valid_ids

    return None


@pytest.mark.asyncio
async def test_confidence_increases_with_more_hits(
    classifier: QueryClassifier,
    account_seizure_metadata: SubAgentMetadata,
) -> None:
    """More keyword hits should produce higher confidence."""

    available: list[SubAgentMetadata] = [account_seizure_metadata]

    # Single keyword hit.
    result_low: ClassificationResult = await classifier.classify_query(
        query="seizure",
        available_domains=available,
    )

    # Multiple keyword hits.
    result_high: ClassificationResult = await classifier.classify_query(
        query="account seizure order protected amounts priority claims",
        available_domains=available,
    )

    assert result_high.confidence >= result_low.confidence

    return None
