"""
Data models for the Legal Advisor System.

This module defines all shared data structures used across the system's layers:
executor, supervisor, sub-agents, registry, and context store. Using dataclasses
from the standard library keeps the models lightweight and serialization-agnostic,
while full typing and docstrings ensure clarity for contributors extending the system
with new sub-agents or capabilities.

The models are organized by their role in the processing pipeline:
- Request models (ChatRequest) — validated input from the client
- Classification models (ClassificationResult) — output of query classification
- Sub-agent models (SubAgentMetadata, SubAgentResponse, SubAgentResult) — registry and response contracts
- Synthesis models (SynthesizedResponse) — merged output from multiple sub-agents
- Context models (Exchange, ConversationContext) — session history management
- Streaming models (StreamChunk) — token-by-token delivery to the client
- Enums (Language, ConfidenceLevel) — constrained value sets for type safety
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Language(Enum):
    """
    Supported query languages for the Legal Advisor System.

    The system restricts input to German and English because the legal knowledge base
    covers German banking law (ZPO, InsO) and the user base consists of German banking
    professionals who may also operate in English. Queries in other languages are rejected
    at the validation layer to avoid unreliable classification or hallucinated legal citations.
    """

    GERMAN = "de"
    ENGLISH = "en"


class ConfidenceLevel(Enum):
    """
    Confidence qualifier for sub-agent responses.

    Confidence levels communicate to the end user how reliable a legal answer is,
    enabling them to decide whether additional professional consultation is needed.
    The three-tier scale maps directly to the source-matching quality:
    - HIGH: directly applicable provision with explicit wording found
    - MEDIUM: answer derived by analogy or from general provisions
    - LOW: no directly matching provision; answer relies on interpretation
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ChatRequest:
    """
    Validated incoming request from the client.

    Represents a fully validated and parsed user query after the executor layer
    has confirmed authentication, payload structure, query length, and language.
    Downstream components (supervisor, sub-agents) can trust that all fields
    satisfy the system's input constraints.

    Attributes:
        query: The user's natural-language legal question (1–2000 characters).
        session_id: Unique identifier for the conversation session, used to
            retrieve and store conversation context across exchanges.
        language: Detected or declared language of the query, constrained to
            the supported set (German, English).
    """

    query: str
    session_id: str
    language: Language


@dataclass
class LegalReference:
    """
    A citation to a specific legal provision.

    Legal references are the core traceability mechanism of the system. Every
    substantive answer must cite at least one provision so that banking professionals
    can verify the information against the original legal text and use it in
    compliance documentation.

    Attributes:
        law_name: Abbreviated name of the law (e.g., "ZPO", "InsO", "PfÜB").
        paragraph: The specific paragraph cited (e.g., "§ 850c", "§ 80").
        section: Optional sub-section for more precise citation (e.g., "Abs. 1").
            Not all provisions require sub-section granularity.
    """

    law_name: str
    paragraph: str
    section: Optional[str] = None


@dataclass
class SubAgentMetadata:
    """
    Metadata describing a sub-agent's capabilities for registry discovery.

    The registry stores this metadata at startup so the supervisor's classifier
    can determine which sub-agents are relevant for a given query without
    invoking the agents themselves. This enables extensibility: adding a new
    sub-agent only requires registering new metadata — no supervisor code changes.

    Attributes:
        domain_id: Unique identifier for the legal sub-domain this agent covers
            (e.g., "account_seizure", "insolvency"). Used as the registry key
            and referenced in ClassificationResult.
        description: Human-readable description of the topics this agent handles.
            Used by the classifier to match queries to domains and shown to users
            when listing available sub-domains.
        supported_categories: Specific query categories this agent can answer
            (e.g., ["seizure_order", "protected_amounts"]). Provides finer-grained
            matching beyond the domain_id level.
    """

    domain_id: str
    description: str
    supported_categories: list[str]


@dataclass
class ClassificationResult:
    """
    Result of query classification by the supervisor.

    After the classifier analyzes a query against all registered sub-agent metadata,
    it produces this result indicating which sub-domains should handle the query.
    An empty domain_ids list signals that no domain matched, triggering the supervisor
    to respond with a clarification listing available sub-domains.

    Attributes:
        domain_ids: List of sub-domain identifiers that the query maps to.
            May contain multiple entries for cross-domain queries (e.g., a question
            about seizure priority during insolvency). Empty if no match found.
        confidence: Numeric confidence score (0.0–1.0) indicating how certain
            the classifier is about the domain assignment.
        language: Detected language of the query, carried forward so downstream
            components can respond in the same language.
    """

    domain_ids: list[str]
    confidence: float
    language: Language


@dataclass
class SubAgentResponse:
    """
    Structured response from a single sub-agent after processing a query.

    This is the standard output contract that all sub-agents must produce.
    The synthesizer relies on the consistent structure to merge responses
    from multiple agents into a coherent final answer.

    Attributes:
        domain_id: Identifies which sub-agent produced this response, enabling
            the synthesizer to organize content by sub-domain.
        answer_body: The substantive legal answer text. May be empty if the
            query is out of scope or unresolvable.
        references: Legal provisions cited in the answer. Must contain at least
            one entry for substantive answers (HIGH/MEDIUM confidence).
            Empty for out-of-scope or unresolvable queries.
        confidence: How reliable this answer is, based on how well the query
            matched available legal provisions.
        is_out_of_scope: True if the query does not fall within this agent's
            covered topics. Signals the supervisor to re-route or inform the user.
        limitation_note: Explanation of what could not be resolved, populated
            when no matching provision is found but the query is within scope.
            Ensures transparency about the system's limitations.
    """

    domain_id: str
    answer_body: str
    references: list[LegalReference]
    confidence: ConfidenceLevel
    is_out_of_scope: bool = False
    limitation_note: Optional[str] = None


@dataclass
class SubAgentResult:
    """
    Wrapper for dispatch results including timeout and error handling.

    The parallel dispatcher produces one SubAgentResult per invoked sub-agent,
    regardless of whether the agent succeeded, timed out, or raised an error.
    This uniform wrapper enables the synthesizer to handle partial failures
    gracefully — successful responses are merged while failed domains are
    reported in the unresolved_domains list.

    Attributes:
        domain_id: Identifies which sub-agent this result corresponds to.
        response: The sub-agent's response if it completed successfully within
            the timeout. None if the agent timed out or errored.
        timed_out: True if the agent did not respond within the 30-second
            per-agent timeout. Mutually exclusive with a successful response.
        error: Error message if the agent raised an exception during processing.
            None for successful completions and timeouts.
    """

    domain_id: str
    response: Optional[SubAgentResponse] = None
    timed_out: bool = False
    error: Optional[str] = None


@dataclass
class SynthesizedResponse:
    """
    Final merged response from the supervisor after combining sub-agent outputs.

    This is the last structured representation before the response is streamed
    to the client. It consolidates answers from multiple sub-agents, preserves
    all legal references without alteration, and flags domains that could not
    be resolved due to timeouts or errors.

    Attributes:
        answer_body: Merged and deduplicated answer text, organized by sub-domain.
        references: Complete set of legal references from all successful sub-agent
            responses. No reference is dropped or altered during synthesis.
        confidence: Overall confidence level for the merged response. Set to the
            lowest confidence among contributing sub-agents to be conservative.
        unresolved_domains: Domain IDs of sub-agents that timed out or errored,
            informing the user which aspects of their query could not be addressed.
        recommend_professional: True if any contributing sub-agent reported LOW
            confidence, signaling that professional legal consultation is advised.
    """

    answer_body: str
    references: list[LegalReference]
    confidence: ConfidenceLevel
    unresolved_domains: list[str] = field(default_factory=list)
    recommend_professional: bool = False


@dataclass
class Exchange:
    """
    A single user-system exchange in a conversation session.

    Represents one turn of dialogue: the user's query and the system's response.
    Exchanges are stored sequentially in the context store to enable follow-up
    questions that reference prior context without repeating background information.

    Attributes:
        user_query: The original query text submitted by the user in this turn.
        system_response: The system's answer text delivered to the user.
        references: Legal references cited in the system's response for this turn.
            Preserved during context summarization as key entities.
    """

    user_query: str
    system_response: str
    references: list[LegalReference] = field(default_factory=list)


@dataclass
class ConversationContext:
    """
    Full conversation context for a session, passed to sub-agents with each query.

    Enables sub-agents to understand prior exchanges and provide contextually
    relevant answers to follow-up questions. When the session exceeds the context
    window limit (20 exchanges), older exchanges are summarized while preserving
    key entities (legal references, case identifiers, monetary amounts, party names).

    Attributes:
        exchanges: Ordered list of all exchanges in the session (or the most recent
            ones after truncation). Sub-agents use this to resolve references to
            prior context in follow-up queries.
        summary: Condensed representation of older exchanges after summarization.
            None if the session has not yet exceeded the context window limit.
        is_truncated: True if context summarization has occurred, signaling to the
            user that earlier context has been condensed and repeating critical
            details may improve answer accuracy.
        preserved_entities: Key entities extracted from summarized exchanges that
            must remain accessible (legal references, case identifiers, monetary
            amounts, party names). Ensures continuity even after truncation.
    """

    exchanges: list[Exchange]
    summary: Optional[str] = None
    is_truncated: bool = False
    preserved_entities: list[str] = field(default_factory=list)


@dataclass
class StreamChunk:
    """
    A single chunk in a streaming SSE response delivered to the client.

    The system streams responses token-by-token to meet the 3-second first-token
    latency requirement. Each chunk carries a content fragment; the final chunk
    is marked with is_final=True and may include metadata (e.g., total references,
    confidence level, unresolved domains) for client-side rendering.

    Attributes:
        content: Text fragment to append to the response being assembled client-side.
            May be a single token, word, or sentence fragment depending on generation.
        is_final: True for the last chunk in the stream. Clients use this to know
            when the response is complete and metadata is available.
        metadata: Optional dictionary attached to the final chunk containing
            structured information about the response (e.g., confidence level,
            reference count, unresolved domains). None for non-final chunks.
    """

    content: str
    is_final: bool = False
    metadata: Optional[dict] = None
