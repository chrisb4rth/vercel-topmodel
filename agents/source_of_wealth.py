"""
Source of Wealth Sub-Agent module.

Implements the specialized sub-agent for answering questions about the source
of wealth (Mittelherkunft) of banking customers. This agent supports KYC/AML
compliance by retrieving background information on subjects from the web via
the Parallel API, then synthesizing findings into a structured response with
references to applicable regulations (GwG — Geldwäschegesetz).

The agent can be used to:
- Research the professional background and business activities of a subject
- Identify publicly available information about wealth origins
- Flag potential inconsistencies or risk indicators
- Cite GwG provisions governing source-of-wealth verification obligations

Configuration:
    Set the environment variable PARALLEL_API_KEY to authenticate with the
    Parallel API. The agent will raise a configuration error if the key is
    not set when a query is processed.
"""

import logging
import os
from typing import Optional

from agents.base import BaseSubAgent
from models import (
    ConfidenceLevel,
    ConversationContext,
    LegalReference,
    SubAgentMetadata,
    SubAgentResponse,
)

logger: logging.Logger = logging.getLogger(__name__)


# Legal references applicable to source-of-wealth verification in German banking.
_SOW_REFERENCES: list[LegalReference] = [
    LegalReference(law_name="GwG", paragraph="§ 10", section="Abs. 1 Nr. 2"),
    LegalReference(law_name="GwG", paragraph="§ 11", section="Abs. 1"),
    LegalReference(law_name="GwG", paragraph="§ 15", section="Abs. 2"),
]

# The SOW research prompt sent to the Parallel API as the task input prefix.
_SOW_RESEARCH_PROMPT: str = (
    "# Source of Wealth (SOW) Research Prompt for Banking Compliance\n\n"
    "## System Prompt\n\n"
    "You are an AI research assistant specialized in conducting source of wealth "
    "verification for banking compliance and know-your-customer (KYC) processes. "
    "Your role is to gather, analyze, and document publicly available information "
    "about individuals to support banks' due diligence obligations under "
    "anti-money laundering (AML) and counter-terrorism financing (CTF) regulations.\n\n"
    "### Key Principles\n\n"
    "1. **Compliance First**: All research must align with banking regulations "
    "(e.g., FATF guidelines, FinCEN, EU 5AMLD, local jurisdictional requirements)\n"
    "2. **Evidence-Based**: Only report information found in publicly available, "
    "credible sources\n"
    "3. **Structured Output**: Deliver findings in a standardized format\n"
    "4. **Risk Assessment**: Contextualize findings within AML/CFT risk frameworks\n"
    "5. **Source Documentation**: Every claim must be traceable to its source\n"
    "6. **Impartiality**: Report findings objectively without bias or speculation\n\n"
    "---\n\n"
    "## Research Directive\n\n"
    "Conduct targeted web research across these categories:\n"
    "1. Professional & Business Background\n"
    "2. Public Business Activities & Wealth Sources\n"
    "3. Financial Indicators\n"
    "4. Negative Indicators & Risk Signals (sanctions, PEP, adverse media)\n"
    "5. Verification & Corroboration\n\n"
    "---\n\n"
    "## Subject Information\n\n"
)


class SourceOfWealthAgent(BaseSubAgent):
    """
    Sub-agent specialized in source-of-wealth research for KYC/AML compliance.

    Uses the Parallel API to conduct web-based research on subjects (customers,
    beneficial owners) and synthesizes findings into a structured response.
    Cites GwG (Geldwäschegesetz) provisions governing the bank's obligation
    to verify the origin of funds.

    The agent reads the API key from the PARALLEL_API_KEY environment variable.
    """

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
        Process a source-of-wealth query by calling the Parallel search API.

        Sends the query to the Parallel API which conducts web research on the
        subject, then formats the results into a structured response with
        applicable GwG references.

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

        # Retrieve the API key from environment.
        api_key: Optional[str] = os.environ.get("PARALLEL_API_KEY")
        if not api_key:
            logger.error(
                "PARALLEL_API_KEY environment variable is not set. "
                "Cannot perform source-of-wealth research."
            )
            limitation_note: str = (
                "Die Recherche konnte nicht durchgeführt werden: "
                "API-Schlüssel nicht konfiguriert."
                if is_german
                else "Research could not be performed: API key not configured."
            )
            return SubAgentResponse(
                domain_id="source_of_wealth",
                answer_body=limitation_note,
                references=list(_SOW_REFERENCES),
                confidence=ConfidenceLevel.LOW,
                is_out_of_scope=False,
                limitation_note=limitation_note,
            )

        # Call the Parallel API to conduct research.
        try:
            research_output: str = await _call_parallel_api(api_key, query)
        except Exception as e:
            logger.error("Parallel API call failed: %s", str(e))
            limitation_note = (
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

        # If the API returned empty or no useful content.
        if not research_output or len(research_output.strip()) < 50:
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

        # Build the final answer with regulatory context.
        answer_body: str = _format_answer(research_output, is_german)
        confidence: ConfidenceLevel = _assess_confidence(research_output)

        return SubAgentResponse(
            domain_id="source_of_wealth",
            answer_body=answer_body,
            references=list(_SOW_REFERENCES),
            confidence=confidence,
            is_out_of_scope=False,
            limitation_note=None,
        )


async def _call_parallel_api(api_key: str, query: str) -> str:
    """
    Call the Parallel API to conduct source-of-wealth research.

    Creates a task run with the SOW research prompt and the user's query,
    then waits for the result.

    Args:
        api_key: The Parallel API key.
        query: The user's research query about a subject.

    Returns:
        The research output text from the Parallel API.

    Raises:
        Exception: If the API call fails or times out.
    """
    import asyncio
    from parallel import Parallel

    # Run the synchronous Parallel client in a thread to avoid blocking
    # the async event loop.
    def _run_task() -> str:
        client = Parallel(api_key=api_key)

        task_input: str = _SOW_RESEARCH_PROMPT + query

        task_run = client.task_run.create(
            input=task_input,
            processor="core-fast",
            task_spec={
                "input_schema": {
                    "type": "text",
                    "description": "The user request to execute.",
                },
                "output_schema": {
                    "type": "text",
                    "description": (
                        "Return a helpful final answer in clear markdown "
                        "that addresses the user request."
                    ),
                },
            },
        )

        run_result = client.task_run.result(task_run.run_id, api_timeout=3600)
        return run_result.output

    loop = asyncio.get_event_loop()
    result: str = await loop.run_in_executor(None, _run_task)
    return result


def _format_answer(research_output: str, is_german: bool) -> str:
    """
    Format the Parallel API research output with regulatory context.

    Args:
        research_output: Raw research text from the Parallel API.
        is_german: Whether to produce a German-language wrapper.

    Returns:
        Formatted answer body string.
    """
    if is_german:
        header = (
            "**Mittelherkunftsprüfung gemäß § 10 Abs. 1 Nr. 2 GwG**\n\n"
            "Die Bank ist verpflichtet, die Herkunft der Mittel zu klären. "
            "Die folgende Recherche ergab:\n\n"
        )
        footer = (
            "\n\n---\n\n"
            "**Hinweis:** Diese Informationen stammen aus öffentlich "
            "zugänglichen Quellen und ersetzen keine vollständige "
            "Sorgfaltsprüfung gemäß § 11 Abs. 1 GwG. Bei erhöhtem Risiko "
            "sind verstärkte Sorgfaltspflichten nach § 15 Abs. 2 GwG "
            "anzuwenden."
        )
    else:
        header = (
            "**Source of Wealth Verification pursuant to § 10(1) No. 2 GwG**\n\n"
            "The bank is obligated to clarify the origin of funds. "
            "The following research was found:\n\n"
        )
        footer = (
            "\n\n---\n\n"
            "**Note:** This information is derived from publicly available "
            "sources and does not replace a full due diligence review pursuant "
            "to § 11(1) GwG. In cases of elevated risk, enhanced due diligence "
            "measures under § 15(2) GwG must be applied."
        )

    return header + research_output + footer


def _assess_confidence(research_output: str) -> ConfidenceLevel:
    """
    Assess confidence level based on the quality of research output.

    Uses simple heuristics on the output length and content indicators
    to determine confidence.

    Args:
        research_output: The research text from the Parallel API.

    Returns:
        HIGH if output is substantial with multiple sources, MEDIUM if
        moderate, LOW if minimal.
    """
    output_length: int = len(research_output)

    # Check for indicators of substantive research.
    has_sources: bool = (
        "source" in research_output.lower()
        or "quelle" in research_output.lower()
        or "http" in research_output.lower()
    )
    has_findings: bool = (
        "found" in research_output.lower()
        or "identified" in research_output.lower()
        or "gefunden" in research_output.lower()
        or "festgestellt" in research_output.lower()
    )

    if output_length > 2000 and has_sources and has_findings:
        return ConfidenceLevel.HIGH
    elif output_length > 500 and (has_sources or has_findings):
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


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
