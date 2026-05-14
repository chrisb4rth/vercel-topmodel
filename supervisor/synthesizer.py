"""
Response synthesizer module for merging sub-agent outputs.

This module implements the ResponseSynthesizer class, which is responsible for
combining multiple SubAgentResult objects (from the parallel dispatcher) into a
single SynthesizedResponse. The synthesizer handles three categories of results:

    1. Successful responses — merged into the answer body, organized by sub-domain.
    2. Out-of-scope responses — excluded from the main answer but noted.
    3. Failed results (timed out or errored) — their domain_ids are collected
       into the unresolved_domains list.

Key invariants maintained during synthesis:
    - All legal references from successful sub-agent responses are preserved
      without alteration (Property 6 / Req 7.4).
    - Duplicate answer content is removed while preserving unique information.
    - The overall confidence level is set to the LOWEST among successful responses
      (conservative approach).
    - recommend_professional is set to True if ANY response has LOW confidence
      (Req 7.3).
    - Unresolved domains are populated for timed-out or errored agents (Req 2.4).

Requirements: 2.2, 2.4, 7.3, 7.4
"""

from models import (
    ConfidenceLevel,
    LegalReference,
    SubAgentResponse,
    SubAgentResult,
    SynthesizedResponse,
)


# Confidence levels ordered from lowest to highest for comparison
_CONFIDENCE_RANK: dict[ConfidenceLevel, int] = {
    ConfidenceLevel.LOW: 0,
    ConfidenceLevel.MEDIUM: 1,
    ConfidenceLevel.HIGH: 2,
}


class ResponseSynthesizer:
    """
    Merges multiple sub-agent dispatch results into a single synthesized response.

    The synthesizer is stateless — each call to `synthesize` processes an
    independent batch of results. It does not cache prior synthesis outputs
    or maintain references to sub-agents between invocations.

    The synthesis pipeline:
        1. Separate successful results from failed ones (timed_out / error).
        2. Collect unresolved_domains from failed results.
        3. Filter out out-of-scope responses from the main answer.
        4. Collect all legal references from successful, in-scope responses
           (preserved without alteration).
        5. Merge answer bodies organized by sub-domain (domain_id as header).
        6. Determine overall confidence as the LOWEST among successful responses.
        7. Set recommend_professional=True if any response has LOW confidence.
        8. Handle edge case: if all results failed, return empty answer with
           all domains as unresolved.

    Usage:
        synthesizer = ResponseSynthesizer()
        response = await synthesizer.synthesize(
            results=dispatch_results,
            query="What is the protected amount under seizure?",
        )
    """

    async def synthesize(
        self,
        results: list[SubAgentResult],
        query: str,
    ) -> SynthesizedResponse:
        """
        Merge dispatch results into a single coherent response.

        Processes the full list of SubAgentResult objects produced by the
        parallel dispatcher, separating successes from failures, and combining
        successful responses into a unified answer with preserved references.

        Args:
            results: List of SubAgentResult instances from the dispatcher.
                Each contains either a successful SubAgentResponse, a timed_out
                flag, or an error message.
            query: The original user query, retained for potential context in
                synthesis (e.g., organizing answer sections by relevance).

        Returns:
            A SynthesizedResponse containing the merged answer body, all
            collected legal references, the conservative confidence level,
            unresolved domain list, and professional consultation flag.
        """

        # Separate failed results (timed_out or errored) from successful ones
        successful_responses: list[SubAgentResponse] = []
        unresolved_domains: list[str] = []

        for result in results:
            if result.response is not None:
                successful_responses.append(result.response)
            else:
                # Agent timed out or raised an error — domain is unresolved
                unresolved_domains.append(result.domain_id)

        # Edge case: all agents failed — return empty answer with all domains unresolved
        if not successful_responses:
            empty_response: SynthesizedResponse = SynthesizedResponse(
                answer_body="",
                references=[],
                confidence=ConfidenceLevel.LOW,
                unresolved_domains=unresolved_domains,
                recommend_professional=True,
            )
            return empty_response

        # Separate in-scope responses from out-of-scope ones
        in_scope_responses: list[SubAgentResponse] = []
        out_of_scope_responses: list[SubAgentResponse] = []

        for response in successful_responses:
            if response.is_out_of_scope:
                out_of_scope_responses.append(response)
            else:
                in_scope_responses.append(response)

        # If all successful responses are out-of-scope, treat as effectively empty
        if not in_scope_responses:
            out_of_scope_answer: SynthesizedResponse = SynthesizedResponse(
                answer_body=_build_out_of_scope_note(out_of_scope_responses),
                references=[],
                confidence=ConfidenceLevel.LOW,
                unresolved_domains=unresolved_domains,
                recommend_professional=True,
            )
            return out_of_scope_answer

        # Collect ALL legal references from ALL successful responses (in-scope
        # and out-of-scope) without alteration — Property 6 / Req 7.4
        all_references: list[LegalReference] = _collect_all_references(
            successful_responses
        )

        # Merge answer bodies organized by sub-domain, removing duplicates
        merged_answer_body: str = _merge_answer_bodies(in_scope_responses)

        # Append out-of-scope notes if any agents flagged the query as outside scope
        if out_of_scope_responses:
            out_of_scope_note: str = _build_out_of_scope_note(out_of_scope_responses)
            merged_answer_body = f"{merged_answer_body}\n\n{out_of_scope_note}"

        # Determine overall confidence: use the LOWEST among successful in-scope
        # responses (conservative approach)
        overall_confidence: ConfidenceLevel = _determine_lowest_confidence(
            in_scope_responses
        )

        # Set recommend_professional if ANY response (in-scope or out-of-scope)
        # has LOW confidence — Req 7.3
        recommend_professional: bool = _any_low_confidence(successful_responses)

        synthesized_response: SynthesizedResponse = SynthesizedResponse(
            answer_body=merged_answer_body,
            references=all_references,
            confidence=overall_confidence,
            unresolved_domains=unresolved_domains,
            recommend_professional=recommend_professional,
        )
        return synthesized_response


def _collect_all_references(
    responses: list[SubAgentResponse],
) -> list[LegalReference]:
    """
    Collect all legal references from all responses without alteration.

    References are preserved exactly as provided by each sub-agent. No
    deduplication is performed on references because even seemingly identical
    citations may carry different contextual significance, and the requirement
    (Req 7.4) explicitly states "without alteration".

    Args:
        responses: List of successful SubAgentResponse instances to extract
            references from.

    Returns:
        A flat list of all LegalReference objects from all responses,
        maintaining the order: first all references from the first response,
        then all from the second, etc.
    """

    all_references: list[LegalReference] = []
    for response in responses:
        for reference in response.references:
            all_references.append(reference)

    return all_references


def _merge_answer_bodies(
    responses: list[SubAgentResponse],
) -> str:
    """
    Merge answer bodies from multiple in-scope responses, organized by sub-domain.

    Each response's answer is placed under a section header using the domain_id.
    Duplicate content (identical answer_body text from different agents) is
    removed to avoid redundancy. Limitation notes are appended to their
    respective domain sections when present.

    Args:
        responses: List of in-scope SubAgentResponse instances to merge.

    Returns:
        A single string containing all unique answer content organized by
        sub-domain sections.
    """

    # Track seen answer bodies to remove exact duplicates
    seen_bodies: set[str] = set()
    sections: list[str] = []

    for response in responses:
        # Skip empty answer bodies
        if not response.answer_body.strip():
            # Still include limitation note if present
            if response.limitation_note:
                section: str = (
                    f"[{response.domain_id}]\n{response.limitation_note}"
                )
                sections.append(section)
            continue

        # Skip duplicate answer bodies
        normalized_body: str = response.answer_body.strip()
        if normalized_body in seen_bodies:
            continue
        seen_bodies.add(normalized_body)

        # Build section with domain header
        section_parts: list[str] = [f"[{response.domain_id}]", normalized_body]

        # Append limitation note if the agent flagged partial limitations
        if response.limitation_note:
            section_parts.append(f"Note: {response.limitation_note}")

        section = "\n".join(section_parts)
        sections.append(section)

    merged_body: str = "\n\n".join(sections)
    return merged_body


def _build_out_of_scope_note(
    out_of_scope_responses: list[SubAgentResponse],
) -> str:
    """
    Build a note summarizing which domains flagged the query as out of scope.

    This informs the user that certain sub-agents could not address their
    query because it falls outside those agents' covered topics.

    Args:
        out_of_scope_responses: List of SubAgentResponse instances where
            is_out_of_scope is True.

    Returns:
        A human-readable note listing the out-of-scope domains.
    """

    domain_ids: list[str] = [r.domain_id for r in out_of_scope_responses]
    domains_text: str = ", ".join(domain_ids)
    note: str = (
        f"The following domains indicated this query is outside their scope: "
        f"{domains_text}."
    )
    return note


def _determine_lowest_confidence(
    responses: list[SubAgentResponse],
) -> ConfidenceLevel:
    """
    Determine the lowest confidence level among a list of responses.

    Uses a conservative approach: the overall confidence is only as strong
    as the weakest contributing response. This ensures users are not
    misled by a high-confidence partial answer when another part of the
    response is uncertain.

    Args:
        responses: List of SubAgentResponse instances to evaluate.
            Must not be empty.

    Returns:
        The ConfidenceLevel with the lowest rank among all responses.
    """

    lowest_confidence: ConfidenceLevel = ConfidenceLevel.HIGH

    for response in responses:
        if _CONFIDENCE_RANK[response.confidence] < _CONFIDENCE_RANK[lowest_confidence]:
            lowest_confidence = response.confidence

    return lowest_confidence


def _any_low_confidence(
    responses: list[SubAgentResponse],
) -> bool:
    """
    Check whether any response in the list has LOW confidence.

    Used to determine whether the synthesized response should recommend
    professional legal consultation (Req 7.3).

    Args:
        responses: List of SubAgentResponse instances to check.

    Returns:
        True if at least one response has ConfidenceLevel.LOW, False otherwise.
    """

    for response in responses:
        if response.confidence == ConfidenceLevel.LOW:
            return True

    return False
