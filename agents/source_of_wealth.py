"""
Source of Wealth Sub-Agent module.

Implements the specialized sub-agent for answering questions about the source
of wealth (Mittelherkunft) of banking customers. This agent supports KYC/AML
compliance by retrieving background information on subjects from the web via
a search API tool, then synthesizing findings into a structured response with
references to applicable regulations (GwG — Geldwäschegesetz).

The agent can be used to:
- Research the professional background and business activities of a subject
- Identify publicly available information about wealth origins
- Flag potential inconsistencies or risk indicators
- Cite GwG provisions governing source-of-wealth verification obligations
"""

import logging
from typing import Callable, Awaitable, Optional

from agents.base import BaseSubAgent
from models import (
    ConfidenceLevel,
    ConversationContext,
    LegalReference,
    SubAgentMetadata,
    SubAgentResponse,
)

logger: logging.Logger = logging.getLogger(__name__)


# Type alias for the search tool function signature.
# The search tool accepts a query string and returns a list of result dicts,
# each containing at minimum "title", "snippet", and "url" keys.
SearchTool = Callable[[str], Awaitable[list[dict]]]


# Legal references applicable to source-of-wealth verification in German banking.
_SOW_REFERENCES: list[LegalReference] = [
    LegalReference(law_name="GwG", paragraph="§ 10", section="Abs. 1 Nr. 2"),
    LegalReference(law_name="GwG", paragraph="§ 11", section="Abs. 1"),
    LegalReference(law_name="GwG", paragraph="§ 15", section="Abs. 2"),
]


class SourceOfWealthAgent(BaseSubAgent):
    """
    Sub-agent specialized in source-of-wealth research for KYC/AML compliance.

    Uses a search API tool to retrieve publicly available background information
    on subjects (customers, beneficial owners) and synthesizes findings into a
    structured response. Cites GwG (Geldwäschegesetz) provisions governing the
    bank's obligation to verify the origin of funds.

    The agent requires a search tool to be injected at construction time. This
    tool is called during query processing to retrieve web-based background
    information on the subject in question.

    Args:
        search_tool: An async callable that accepts a search query string and
            returns a list of result dictionaries with "title", "snippet", and
            "url" keys. This is the agent's primary mechanism for retrieving
            external information.
    """

    def __init__(self, search_tool: SearchTool) -> None:
        """
        Initialize the Source of Wealth agent with a search tool.

        Args:
            search_tool: Async callable that performs web searches. Must accept
                a query string and return a list of dicts with keys "title",
                "snippet", and "url".
        """
        self._search_tool: SearchTool = search_tool

    def get_metadata(self) -> SubAgentMetadata:
        """
        Return metadata describing the source-of-wealth agent's capabilities.

        Returns:
            SubAgentMetadata with domain_id "source_of_wealth", a description
            of covered topics, and the list of supported query categories.
        """
        return SubAgentMetadata(
            domain_id="source_of_wealth",
            description=(
                "Handles source-of-wealth (Mittelherkunft) research for "
                "KYC/AML compliance in German banking operations. Retrieves "
                "background information on subjects from the web to support "
                "verification of the origin of funds, citing GwG provisions."
            ),
            supported_categories=[
                "wealth_origin_research",
                "beneficial_owner_background",
                "pep_screening",
                "adverse_media_check",
            ],
        )

    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """
        Process a source-of-wealth query by searching for background information.

        Calls the injected search tool to retrieve publicly available information
        about the subject, then synthesizes the results into a structured response
        with applicable GwG references.

        Args:
            query: The user's natural-language question about a subject's source
                of wealth or background, already classified as belonging to this
                domain.
            context: Conversation history for the current session.

        Returns:
            SubAgentResponse with synthesized background information, GwG
            references, confidence level, and scope/limitation flags.
        """
        query_lower: str = query.lower()

        # Determine language from query content.
        is_german: bool = _detect_german(query_lower)

        # Check if the query is within this agent's domain.
        if not _is_within_domain(query_lower):
            return SubAgentResponse(
                domain_id="source_of_wealth",
                answer_body="",
                references=[],
                confidence=ConfidenceLevel.LOW,
                is_out_of_scope=True,
                limitation_note=None,
            )

        # Build a search query from the user's question.
        search_query: str = _build_search_query(query)

        # Call the search tool to retrieve background information.
        try:
            search_results: list[dict] = await self._search_tool(search_query)
        except Exception as e:
            logger.error("Search tool failed: %s", str(e))
            limitation_note: str = (
                "Die Suche nach Hintergrundinformationen konnte nicht "
                "durchgeführt werden. Bitte versuchen Sie es erneut."
                if is_german
                else "The background information search could not be completed. "
                "Please try again."
            )
            return SubAgentResponse(
                domain_id="source_of_wealth",
                answer_body=limitation_note,
                references=list(_SOW_REFERENCES),
                confidence=ConfidenceLevel.LOW,
                is_out_of_scope=False,
                limitation_note=limitation_note,
            )

        # If no results were found, return a low-confidence response.
        if not search_results:
            limitation_note = (
                "Zu der angefragten Person/Entität konnten keine öffentlich "
                "verfügbaren Informationen zur Mittelherkunft gefunden werden. "
                "Eine erweiterte Prüfung gemäß § 15 Abs. 2 GwG wird empfohlen."
                if is_german
                else "No publicly available information on the source of wealth "
                "could be found for the requested subject. Enhanced due diligence "
                "pursuant to § 15(2) GwG is recommended."
            )
            return SubAgentResponse(
                domain_id="source_of_wealth",
                answer_body=limitation_note,
                references=list(_SOW_REFERENCES),
                confidence=ConfidenceLevel.LOW,
                is_out_of_scope=False,
                limitation_note=limitation_note,
            )

        # Synthesize search results into a structured answer.
        answer_body: str = _synthesize_results(search_results, is_german)
        confidence: ConfidenceLevel = _assess_confidence(search_results)

        return SubAgentResponse(
            domain_id="source_of_wealth",
            answer_body=answer_body,
            references=list(_SOW_REFERENCES),
            confidence=confidence,
            is_out_of_scope=False,
            limitation_note=None,
        )


def _detect_german(query_lower: str) -> bool:
    """
    Simple heuristic to detect whether a query is in German.

    Args:
        query_lower: Lowercased query string.

    Returns:
        True if the query appears to be in German.
    """
    german_indicators: list[str] = [
        "mittelherkunft",
        "vermögen",
        "herkunft",
        "kunde",
        "geldwäsche",
        "prüfung",
        "hintergrund",
        "recherche",
        "wie",
        "was",
        "wer",
        "welche",
        "ist",
        "der",
        "die",
        "das",
        "ein",
        "eine",
        "und",
        "oder",
        "bei",
        "für",
        "über",
        "nach",
        "ä",
        "ö",
        "ü",
        "ß",
    ]
    matches: int = sum(1 for word in german_indicators if word in query_lower)
    return matches >= 2


def _is_within_domain(query_lower: str) -> bool:
    """
    Determine whether a query is within the source-of-wealth domain.

    Args:
        query_lower: Lowercased query string.

    Returns:
        True if the query appears to be about source of wealth or KYC/AML.
    """
    domain_indicators: list[str] = [
        "source of wealth",
        "source of funds",
        "mittelherkunft",
        "vermögensherkunft",
        "herkunft der mittel",
        "herkunft des vermögens",
        "kyc",
        "know your customer",
        "aml",
        "anti-money laundering",
        "geldwäsche",
        "beneficial owner",
        "wirtschaftlich berechtigter",
        "pep",
        "politically exposed",
        "politisch exponiert",
        "background check",
        "hintergrundprüfung",
        "due diligence",
        "sorgfaltspflicht",
        "adverse media",
        "negative berichterstattung",
        "wealth",
        "vermögen",
        "recherche",
        "research",
    ]
    return any(indicator in query_lower for indicator in domain_indicators)


def _build_search_query(query: str) -> str:
    """
    Build an optimized search query from the user's question.

    Extracts the core subject and appends relevant context terms to improve
    search result quality for source-of-wealth research.

    Args:
        query: The original user query.

    Returns:
        An optimized search string for the search tool.
    """
    # Use the query as-is but append context for better results.
    # In a production system this would use NLP to extract the subject name.
    return f"{query} background business activities wealth"


def _synthesize_results(search_results: list[dict], is_german: bool) -> str:
    """
    Synthesize search results into a structured answer body.

    Combines titles and snippets from search results into a coherent
    summary, prefixed with the regulatory context.

    Args:
        search_results: List of result dicts with "title", "snippet", "url".
        is_german: Whether to produce a German-language response.

    Returns:
        Formatted answer body string.
    """
    if is_german:
        header = (
            "Gemäß § 10 Abs. 1 Nr. 2 GwG ist die Bank verpflichtet, die "
            "Herkunft der Mittel zu klären. Die folgende Recherche ergab:\n\n"
        )
    else:
        header = (
            "Pursuant to § 10(1) No. 2 GwG, the bank is obligated to clarify "
            "the origin of funds. The following research was found:\n\n"
        )

    findings: list[str] = []
    for i, result in enumerate(search_results[:5], start=1):
        title: str = result.get("title", "Untitled")
        snippet: str = result.get("snippet", "")
        url: str = result.get("url", "")
        findings.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")

    body: str = header + "\n\n".join(findings)

    if is_german:
        body += (
            "\n\n**Hinweis:** Diese Informationen stammen aus öffentlich "
            "zugänglichen Quellen und ersetzen keine vollständige "
            "Sorgfaltsprüfung gemäß § 11 Abs. 1 GwG."
        )
    else:
        body += (
            "\n\n**Note:** This information is derived from publicly available "
            "sources and does not replace a full due diligence review pursuant "
            "to § 11(1) GwG."
        )

    return body


def _assess_confidence(search_results: list[dict]) -> ConfidenceLevel:
    """
    Assess confidence level based on the quality and quantity of search results.

    Args:
        search_results: List of result dicts from the search tool.

    Returns:
        HIGH if 3+ results with substantive snippets, MEDIUM if 1-2 results,
        LOW if results lack substance.
    """
    substantive_results: int = sum(
        1
        for r in search_results
        if len(r.get("snippet", "")) > 50
    )

    if substantive_results >= 3:
        return ConfidenceLevel.HIGH
    elif substantive_results >= 1:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW
