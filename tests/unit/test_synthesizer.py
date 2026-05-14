"""
Unit tests for the ResponseSynthesizer.

Covers the key synthesis behaviors:
    1. Single response — straightforward pass-through with domain header.
    2. Multiple responses — merged by sub-domain with deduplication.
    3. Reference preservation — all references from all agents are kept intact.
    4. Confidence downgrade — overall confidence is the lowest among responses.
    5. Recommend professional — triggered when any response has LOW confidence.
    6. Unresolved domains — populated for timed-out or errored agents.
    7. All-failed scenario — empty answer with all domains unresolved.
    8. Out-of-scope handling — excluded from main answer, noted separately.

Each test uses lightweight SubAgentResult/SubAgentResponse instances constructed
directly from the data models without external dependencies.
"""

import pytest

from models import (
    ConfidenceLevel,
    LegalReference,
    SubAgentResponse,
    SubAgentResult,
    SynthesizedResponse,
)
from supervisor.synthesizer import ResponseSynthesizer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthesizer() -> ResponseSynthesizer:
    """Create a fresh ResponseSynthesizer instance."""

    return ResponseSynthesizer()


def _make_response(
    domain_id: str = "test_domain",
    answer_body: str = "Test answer.",
    references: list[LegalReference] | None = None,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    is_out_of_scope: bool = False,
    limitation_note: str | None = None,
) -> SubAgentResponse:
    """
    Helper to construct a SubAgentResponse with sensible defaults.

    Reduces boilerplate in individual tests by providing default values
    for all fields while allowing selective overrides.
    """

    if references is None:
        references = [LegalReference(law_name="ZPO", paragraph="§ 850c")]

    return SubAgentResponse(
        domain_id=domain_id,
        answer_body=answer_body,
        references=references,
        confidence=confidence,
        is_out_of_scope=is_out_of_scope,
        limitation_note=limitation_note,
    )


def _make_success_result(
    response: SubAgentResponse,
) -> SubAgentResult:
    """Wrap a SubAgentResponse in a successful SubAgentResult."""

    return SubAgentResult(
        domain_id=response.domain_id,
        response=response,
    )


def _make_timeout_result(
    domain_id: str,
) -> SubAgentResult:
    """Create a timed-out SubAgentResult for the given domain."""

    return SubAgentResult(
        domain_id=domain_id,
        timed_out=True,
    )


def _make_error_result(
    domain_id: str,
    error: str = "Agent crashed",
) -> SubAgentResult:
    """Create an errored SubAgentResult for the given domain."""

    return SubAgentResult(
        domain_id=domain_id,
        error=error,
    )


# ---------------------------------------------------------------------------
# Test: Single response synthesis
# ---------------------------------------------------------------------------


class TestSingleResponse:
    """Verify synthesis with a single successful sub-agent response."""

    async def test_single_response_preserves_answer(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """A single successful response is passed through with domain header."""

        response: SubAgentResponse = _make_response(
            domain_id="account_seizure",
            answer_body="The protected amount is defined in § 850c ZPO.",
        )
        results: list[SubAgentResult] = [_make_success_result(response)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="What is the protected amount?",
        )

        assert "[account_seizure]" in synthesized.answer_body
        assert "The protected amount is defined in § 850c ZPO." in synthesized.answer_body

    async def test_single_response_confidence_preserved(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """A single response's confidence becomes the overall confidence."""

        response: SubAgentResponse = _make_response(
            confidence=ConfidenceLevel.MEDIUM,
        )
        results: list[SubAgentResult] = [_make_success_result(response)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Test query",
        )

        assert synthesized.confidence == ConfidenceLevel.MEDIUM

    async def test_single_response_no_unresolved_domains(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """A single successful response yields no unresolved domains."""

        response: SubAgentResponse = _make_response()
        results: list[SubAgentResult] = [_make_success_result(response)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Test query",
        )

        assert synthesized.unresolved_domains == []


# ---------------------------------------------------------------------------
# Test: Multiple responses synthesis
# ---------------------------------------------------------------------------


class TestMultipleResponses:
    """Verify synthesis with multiple successful sub-agent responses."""

    async def test_multiple_responses_organized_by_domain(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Multiple responses are organized with domain headers."""

        response_a: SubAgentResponse = _make_response(
            domain_id="account_seizure",
            answer_body="Seizure answer.",
        )
        response_b: SubAgentResponse = _make_response(
            domain_id="insolvency",
            answer_body="Insolvency answer.",
            references=[LegalReference(law_name="InsO", paragraph="§ 80")],
        )
        results: list[SubAgentResult] = [
            _make_success_result(response_a),
            _make_success_result(response_b),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Cross-domain query",
        )

        assert "[account_seizure]" in synthesized.answer_body
        assert "[insolvency]" in synthesized.answer_body
        assert "Seizure answer." in synthesized.answer_body
        assert "Insolvency answer." in synthesized.answer_body

    async def test_duplicate_content_removed(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Identical answer bodies from different agents are deduplicated."""

        duplicate_body: str = "The same answer from both agents."
        response_a: SubAgentResponse = _make_response(
            domain_id="domain_a",
            answer_body=duplicate_body,
        )
        response_b: SubAgentResponse = _make_response(
            domain_id="domain_b",
            answer_body=duplicate_body,
        )
        results: list[SubAgentResult] = [
            _make_success_result(response_a),
            _make_success_result(response_b),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Duplicate test",
        )

        # The duplicate body should appear only once
        count: int = synthesized.answer_body.count(duplicate_body)
        assert count == 1


# ---------------------------------------------------------------------------
# Test: Reference preservation
# ---------------------------------------------------------------------------


class TestReferencePreservation:
    """Verify that all legal references are preserved without alteration."""

    async def test_all_references_collected(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """References from all successful responses are included in output."""

        ref_a: LegalReference = LegalReference(
            law_name="ZPO", paragraph="§ 850c", section="Abs. 1"
        )
        ref_b: LegalReference = LegalReference(
            law_name="InsO", paragraph="§ 80"
        )
        ref_c: LegalReference = LegalReference(
            law_name="InsO", paragraph="§ 89", section="Abs. 2"
        )

        response_a: SubAgentResponse = _make_response(
            domain_id="seizure",
            references=[ref_a],
        )
        response_b: SubAgentResponse = _make_response(
            domain_id="insolvency",
            references=[ref_b, ref_c],
        )
        results: list[SubAgentResult] = [
            _make_success_result(response_a),
            _make_success_result(response_b),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Reference test",
        )

        # All three references must be present
        assert len(synthesized.references) == 3
        assert ref_a in synthesized.references
        assert ref_b in synthesized.references
        assert ref_c in synthesized.references

    async def test_references_not_altered(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Each reference retains its exact law_name, paragraph, and section."""

        original_ref: LegalReference = LegalReference(
            law_name="PfÜB", paragraph="§ 3", section="Abs. 4 Satz 2"
        )
        response: SubAgentResponse = _make_response(
            references=[original_ref],
        )
        results: list[SubAgentResult] = [_make_success_result(response)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Alteration test",
        )

        assert len(synthesized.references) == 1
        preserved_ref: LegalReference = synthesized.references[0]
        assert preserved_ref.law_name == "PfÜB"
        assert preserved_ref.paragraph == "§ 3"
        assert preserved_ref.section == "Abs. 4 Satz 2"

    async def test_references_from_failed_agents_not_included(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Timed-out or errored agents contribute no references (they have none)."""

        response: SubAgentResponse = _make_response(
            domain_id="good_domain",
            references=[LegalReference(law_name="ZPO", paragraph="§ 850c")],
        )
        results: list[SubAgentResult] = [
            _make_success_result(response),
            _make_timeout_result("bad_domain"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Partial failure",
        )

        # Only the successful agent's reference is present
        assert len(synthesized.references) == 1
        assert synthesized.references[0].law_name == "ZPO"


# ---------------------------------------------------------------------------
# Test: Confidence downgrade
# ---------------------------------------------------------------------------


class TestConfidenceDowngrade:
    """Verify that overall confidence is the lowest among successful responses."""

    async def test_high_and_medium_yields_medium(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Mixing HIGH and MEDIUM confidence yields MEDIUM overall."""

        response_high: SubAgentResponse = _make_response(
            domain_id="domain_a",
            confidence=ConfidenceLevel.HIGH,
        )
        response_medium: SubAgentResponse = _make_response(
            domain_id="domain_b",
            confidence=ConfidenceLevel.MEDIUM,
        )
        results: list[SubAgentResult] = [
            _make_success_result(response_high),
            _make_success_result(response_medium),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Confidence test",
        )

        assert synthesized.confidence == ConfidenceLevel.MEDIUM

    async def test_high_and_low_yields_low(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Mixing HIGH and LOW confidence yields LOW overall."""

        response_high: SubAgentResponse = _make_response(
            domain_id="domain_a",
            confidence=ConfidenceLevel.HIGH,
        )
        response_low: SubAgentResponse = _make_response(
            domain_id="domain_b",
            confidence=ConfidenceLevel.LOW,
        )
        results: list[SubAgentResult] = [
            _make_success_result(response_high),
            _make_success_result(response_low),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Confidence test",
        )

        assert synthesized.confidence == ConfidenceLevel.LOW

    async def test_all_high_yields_high(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """All HIGH confidence responses yield HIGH overall."""

        results: list[SubAgentResult] = [
            _make_success_result(_make_response(domain_id="a", confidence=ConfidenceLevel.HIGH)),
            _make_success_result(_make_response(domain_id="b", confidence=ConfidenceLevel.HIGH)),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="All high",
        )

        assert synthesized.confidence == ConfidenceLevel.HIGH


# ---------------------------------------------------------------------------
# Test: Recommend professional consultation
# ---------------------------------------------------------------------------


class TestRecommendProfessional:
    """Verify that recommend_professional is set correctly based on confidence."""

    async def test_low_confidence_triggers_recommendation(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Any LOW confidence response sets recommend_professional=True."""

        response: SubAgentResponse = _make_response(
            confidence=ConfidenceLevel.LOW,
        )
        results: list[SubAgentResult] = [_make_success_result(response)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Low confidence query",
        )

        assert synthesized.recommend_professional is True

    async def test_medium_confidence_no_recommendation(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """MEDIUM confidence alone does NOT trigger recommendation."""

        response: SubAgentResponse = _make_response(
            confidence=ConfidenceLevel.MEDIUM,
        )
        results: list[SubAgentResult] = [_make_success_result(response)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Medium confidence query",
        )

        assert synthesized.recommend_professional is False

    async def test_high_confidence_no_recommendation(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """HIGH confidence does NOT trigger recommendation."""

        response: SubAgentResponse = _make_response(
            confidence=ConfidenceLevel.HIGH,
        )
        results: list[SubAgentResult] = [_make_success_result(response)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="High confidence query",
        )

        assert synthesized.recommend_professional is False

    async def test_mixed_with_one_low_triggers_recommendation(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Even one LOW among multiple HIGH responses triggers recommendation."""

        results: list[SubAgentResult] = [
            _make_success_result(_make_response(domain_id="a", confidence=ConfidenceLevel.HIGH)),
            _make_success_result(_make_response(domain_id="b", confidence=ConfidenceLevel.LOW)),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Mixed confidence",
        )

        assert synthesized.recommend_professional is True


# ---------------------------------------------------------------------------
# Test: Unresolved domains
# ---------------------------------------------------------------------------


class TestUnresolvedDomains:
    """Verify that timed-out and errored agents populate unresolved_domains."""

    async def test_timeout_adds_to_unresolved(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """A timed-out agent's domain_id appears in unresolved_domains."""

        results: list[SubAgentResult] = [
            _make_success_result(_make_response(domain_id="good_domain")),
            _make_timeout_result("timed_out_domain"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Partial timeout",
        )

        assert "timed_out_domain" in synthesized.unresolved_domains
        assert "good_domain" not in synthesized.unresolved_domains

    async def test_error_adds_to_unresolved(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """An errored agent's domain_id appears in unresolved_domains."""

        results: list[SubAgentResult] = [
            _make_success_result(_make_response(domain_id="good_domain")),
            _make_error_result("errored_domain"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Partial error",
        )

        assert "errored_domain" in synthesized.unresolved_domains

    async def test_multiple_failures_all_listed(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Multiple failed agents all appear in unresolved_domains."""

        results: list[SubAgentResult] = [
            _make_success_result(_make_response(domain_id="good")),
            _make_timeout_result("timeout_a"),
            _make_error_result("error_b"),
            _make_timeout_result("timeout_c"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Multiple failures",
        )

        assert set(synthesized.unresolved_domains) == {
            "timeout_a", "error_b", "timeout_c"
        }


# ---------------------------------------------------------------------------
# Test: All-failed scenario
# ---------------------------------------------------------------------------


class TestAllFailed:
    """Verify behavior when all dispatched agents fail."""

    async def test_all_timeout_returns_empty_answer(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """When all agents time out, answer_body is empty."""

        results: list[SubAgentResult] = [
            _make_timeout_result("domain_a"),
            _make_timeout_result("domain_b"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="All failed",
        )

        assert synthesized.answer_body == ""

    async def test_all_failed_lists_all_unresolved(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """When all agents fail, all domain_ids are in unresolved_domains."""

        results: list[SubAgentResult] = [
            _make_timeout_result("domain_a"),
            _make_error_result("domain_b"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="All failed",
        )

        assert set(synthesized.unresolved_domains) == {"domain_a", "domain_b"}

    async def test_all_failed_confidence_is_low(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """When all agents fail, confidence defaults to LOW."""

        results: list[SubAgentResult] = [
            _make_timeout_result("domain_a"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="All failed",
        )

        assert synthesized.confidence == ConfidenceLevel.LOW

    async def test_all_failed_recommends_professional(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """When all agents fail, recommend_professional is True."""

        results: list[SubAgentResult] = [
            _make_timeout_result("domain_a"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="All failed",
        )

        assert synthesized.recommend_professional is True

    async def test_all_failed_no_references(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """When all agents fail, references list is empty."""

        results: list[SubAgentResult] = [
            _make_error_result("domain_a"),
            _make_error_result("domain_b"),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="All failed",
        )

        assert synthesized.references == []


# ---------------------------------------------------------------------------
# Test: Out-of-scope handling
# ---------------------------------------------------------------------------


class TestOutOfScopeHandling:
    """Verify that out-of-scope responses are handled correctly."""

    async def test_out_of_scope_excluded_from_main_answer(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """Out-of-scope responses do not contribute their answer_body to main content."""

        in_scope: SubAgentResponse = _make_response(
            domain_id="seizure",
            answer_body="Relevant seizure answer.",
        )
        out_of_scope: SubAgentResponse = _make_response(
            domain_id="insolvency",
            answer_body="This is out of scope.",
            is_out_of_scope=True,
        )
        results: list[SubAgentResult] = [
            _make_success_result(in_scope),
            _make_success_result(out_of_scope),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Mixed scope query",
        )

        # In-scope answer is present
        assert "Relevant seizure answer." in synthesized.answer_body
        # Out-of-scope answer body is NOT in the main domain sections
        assert "This is out of scope." not in synthesized.answer_body
        # But a note about out-of-scope domains is included
        assert "insolvency" in synthesized.answer_body
        assert "outside their scope" in synthesized.answer_body

    async def test_all_out_of_scope_returns_note(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """When all responses are out-of-scope, a note is returned."""

        out_of_scope: SubAgentResponse = _make_response(
            domain_id="insolvency",
            answer_body="Not my area.",
            is_out_of_scope=True,
        )
        results: list[SubAgentResult] = [_make_success_result(out_of_scope)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Out of scope query",
        )

        assert "insolvency" in synthesized.answer_body
        assert "outside their scope" in synthesized.answer_body
        assert synthesized.confidence == ConfidenceLevel.LOW
        assert synthesized.recommend_professional is True

    async def test_out_of_scope_references_still_preserved(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """References from out-of-scope responses are still preserved (Property 6)."""

        oos_ref: LegalReference = LegalReference(
            law_name="InsO", paragraph="§ 80"
        )
        in_scope: SubAgentResponse = _make_response(
            domain_id="seizure",
            references=[LegalReference(law_name="ZPO", paragraph="§ 850c")],
        )
        out_of_scope: SubAgentResponse = _make_response(
            domain_id="insolvency",
            references=[oos_ref],
            is_out_of_scope=True,
        )
        results: list[SubAgentResult] = [
            _make_success_result(in_scope),
            _make_success_result(out_of_scope),
        ]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Reference preservation across scope",
        )

        # Both references are preserved — even from out-of-scope agent
        assert len(synthesized.references) == 2
        assert oos_ref in synthesized.references


# ---------------------------------------------------------------------------
# Test: Limitation notes
# ---------------------------------------------------------------------------


class TestLimitationNotes:
    """Verify that limitation notes are included in the synthesized answer."""

    async def test_limitation_note_included_in_section(
        self,
        synthesizer: ResponseSynthesizer,
    ) -> None:
        """A response with a limitation_note includes it in the domain section."""

        response: SubAgentResponse = _make_response(
            domain_id="seizure",
            answer_body="Partial answer.",
            limitation_note="Could not determine the exact threshold.",
        )
        results: list[SubAgentResult] = [_make_success_result(response)]

        synthesized: SynthesizedResponse = await synthesizer.synthesize(
            results=results,
            query="Limitation test",
        )

        assert "Could not determine the exact threshold." in synthesized.answer_body
