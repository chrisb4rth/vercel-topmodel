"""
Context Store module.

Provides the ContextStore class responsible for managing per-session conversation
history. Each session maintains an ordered list of exchanges (user query + system
response pairs) that sub-agents can reference when processing follow-up questions.

When a session exceeds the context window limit (MAX_EXCHANGES_BEFORE_TRUNCATION),
older exchanges are summarized into a condensed text representation while preserving
key entities: legal references (law names, paragraphs), case identifiers, monetary
amounts (EUR/€), and party names. This ensures continuity of legal context even
after truncation.

The store uses an in-memory dictionary keyed by session_id. This is suitable for
single-process deployments; a production system would back this with a persistent
store (Redis, database) for multi-instance deployments.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import logging
import re

from models import ConversationContext, Exchange, LegalReference

# Module-level logger for context store operations.
logger: logging.Logger = logging.getLogger(__name__)

# Maximum number of exchanges a session can hold before summarization triggers.
# Requirement 5.3: minimum 20 exchanges before context truncation occurs.
MAX_EXCHANGES_BEFORE_TRUNCATION: int = 20

# Number of recent exchanges to retain after summarization.
# Keeping the last 10 provides sufficient immediate context for follow-up questions.
EXCHANGES_TO_RETAIN: int = 10

# Regex pattern for monetary amounts in EUR format (e.g., "€1.234,56", "EUR 500",
# "1.000,00 EUR", "1000 €"). Covers common German and international EUR notations.
_MONETARY_PATTERN: re.Pattern[str] = re.compile(
    r"(?:€\s*[\d.,]+|[\d.,]+\s*€|EUR\s*[\d.,]+|[\d.,]+\s*EUR)",
    re.IGNORECASE,
)

# Regex pattern for capitalized multi-word names that likely represent party names
# (e.g., "Deutsche Bank AG", "Max Mustermann"). Requires at least two consecutive
# capitalized words to reduce false positives from sentence-initial capitalization.
_PARTY_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+\b"
)


class ContextStore:
    """
    Manages per-session conversation history with automatic summarization.

    The store maintains an in-memory dictionary of ConversationContext objects,
    one per session_id. It provides three core operations:

        1. get_context: Retrieve (or create) the conversation context for a session.
        2. append_exchange: Add a new user-system exchange to a session's history.
        3. summarize_if_needed: Trigger summarization when the exchange count exceeds
           the configured threshold, preserving key legal entities.

    Summarization condenses older exchanges into a text summary while extracting
    and preserving entities that are critical for legal continuity: law references,
    monetary amounts, case identifiers, and party names.

    Attributes:
        _sessions: Internal dictionary mapping session_id strings to their
            corresponding ConversationContext instances.
    """

    def __init__(self) -> None:
        """
        Initialize an empty context store.

        Sessions are created lazily on first access via get_context().
        """

        # Internal storage: session_id → ConversationContext instance.
        self._sessions: dict[str, ConversationContext] = {}

        return None

    async def get_context(
        self,
        session_id: str,
    ) -> ConversationContext:
        """
        Retrieve conversation history for a session.

        If the session does not yet exist, creates a new empty ConversationContext
        and stores it for future access. This lazy initialization avoids requiring
        explicit session creation calls.

        Args:
            session_id: Unique identifier for the conversation session. Used as
                the dictionary key for context lookup and storage.

        Returns:
            The ConversationContext for the given session, containing all exchanges
            (or the most recent ones after truncation), any summary text, and
            preserved entities.
        """

        # Create a new empty context if this session has not been seen before.
        if session_id not in self._sessions:
            empty_context: ConversationContext = ConversationContext(exchanges=[])
            self._sessions[session_id] = empty_context
            logger.debug("Created new context for session '%s'.", session_id)

        context: ConversationContext = self._sessions[session_id]

        return context

    async def append_exchange(
        self,
        session_id: str,
        exchange: Exchange,
    ) -> None:
        """
        Add a new exchange to the session history.

        Retrieves (or creates) the session context and appends the exchange to
        the end of the exchanges list. The exchange is stored as-is without
        modification — summarization is handled separately via summarize_if_needed().

        Args:
            session_id: Unique identifier for the conversation session.
            exchange: The Exchange instance containing the user query, system
                response, and any legal references cited in the response.

        Returns:
            None
        """

        # Ensure the session context exists before appending.
        context: ConversationContext = await self.get_context(session_id)

        # Append the new exchange to the end of the history.
        context.exchanges.append(exchange)
        logger.debug(
            "Appended exchange to session '%s' (total: %d).",
            session_id,
            len(context.exchanges),
        )

        return None

    async def summarize_if_needed(
        self,
        session_id: str,
    ) -> bool:
        """
        Summarize older exchanges if context window limit is reached.

        Checks whether the session's exchange count exceeds
        MAX_EXCHANGES_BEFORE_TRUNCATION. If so, splits the exchanges into
        "older" (to be summarized) and "recent" (to be retained), extracts
        key entities from the older exchanges, generates a text summary, and
        updates the context accordingly.

        After summarization:
            - exchanges contains only the most recent EXCHANGES_TO_RETAIN entries
            - summary contains a condensed text of the older exchanges
            - is_truncated is set to True
            - preserved_entities contains extracted legal references, monetary
              amounts, case identifiers, and party names

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            True if summarization was performed (exchange count exceeded threshold),
            False if the session is still within the context window limit.
        """

        context: ConversationContext = await self.get_context(session_id)

        # Only summarize when the threshold is exceeded.
        if len(context.exchanges) <= MAX_EXCHANGES_BEFORE_TRUNCATION:
            return False

        logger.info(
            "Session '%s' has %d exchanges (threshold: %d). Summarizing.",
            session_id,
            len(context.exchanges),
            MAX_EXCHANGES_BEFORE_TRUNCATION,
        )

        # Split exchanges into older (to summarize) and recent (to retain).
        split_index: int = len(context.exchanges) - EXCHANGES_TO_RETAIN
        older_exchanges: list[Exchange] = context.exchanges[:split_index]
        recent_exchanges: list[Exchange] = context.exchanges[split_index:]

        # Extract key entities from the older exchanges before discarding them.
        preserved_entities: list[str] = _extract_entities(older_exchanges)

        # Generate a text summary of the older exchanges.
        summary_text: str = _generate_summary(older_exchanges)

        # If there was a previous summary, prepend it to maintain continuity.
        if context.summary:
            summary_text = context.summary + "\n\n" + summary_text

        # Update the context with summarization results.
        context.exchanges = recent_exchanges
        context.summary = summary_text
        context.is_truncated = True

        # Merge new entities with any previously preserved entities, deduplicating.
        existing_entities: set[str] = set(context.preserved_entities)
        for entity in preserved_entities:
            if entity not in existing_entities:
                context.preserved_entities.append(entity)
                existing_entities.add(entity)

        logger.info(
            "Summarization complete for session '%s'. "
            "Retained %d exchanges, preserved %d entities.",
            session_id,
            len(context.exchanges),
            len(context.preserved_entities),
        )

        return True


def _extract_entities(
    exchanges: list[Exchange],
) -> list[str]:
    """
    Extract key entities from a list of exchanges for preservation.

    Scans both user queries and system responses for:
        - Legal references (from Exchange.references field)
        - Monetary amounts (EUR/€ patterns via regex)
        - Party names (capitalized multi-word sequences)

    These entities are critical for maintaining legal context continuity
    after summarization, as sub-agents may need to reference them in
    follow-up answers.

    Args:
        exchanges: The list of Exchange instances to extract entities from.

    Returns:
        A deduplicated list of entity strings found across all exchanges.
    """

    entities: list[str] = []
    seen: set[str] = set()

    for exchange in exchanges:
        # Extract legal references from the structured references field.
        for ref in exchange.references:
            ref_string: str = _format_legal_reference(ref)
            if ref_string not in seen:
                entities.append(ref_string)
                seen.add(ref_string)

        # Scan both query and response text for monetary amounts and party names.
        combined_text: str = exchange.user_query + " " + exchange.system_response

        # Extract monetary amounts (e.g., "€1.234,56", "EUR 500").
        monetary_matches: list[str] = _MONETARY_PATTERN.findall(combined_text)
        for amount in monetary_matches:
            normalized_amount: str = amount.strip()
            if normalized_amount not in seen:
                entities.append(normalized_amount)
                seen.add(normalized_amount)

        # Extract potential party names (capitalized multi-word sequences).
        name_matches: list[str] = _PARTY_NAME_PATTERN.findall(combined_text)
        for name in name_matches:
            if name not in seen:
                entities.append(name)
                seen.add(name)

    return entities


def _format_legal_reference(
    ref: LegalReference,
) -> str:
    """
    Format a LegalReference into a human-readable citation string.

    Produces strings like "§ 850c ZPO" or "§ 80 Abs. 1 InsO" depending
    on whether a section is specified.

    Args:
        ref: The LegalReference instance to format.

    Returns:
        A formatted citation string combining paragraph, optional section,
        and law name.
    """

    if ref.section:
        formatted: str = f"{ref.paragraph} {ref.section} {ref.law_name}"
    else:
        formatted = f"{ref.paragraph} {ref.law_name}"

    return formatted


def _generate_summary(
    exchanges: list[Exchange],
) -> str:
    """
    Generate a condensed text summary of a list of exchanges.

    Creates a structured summary that captures the topics discussed and
    key points from each exchange. The summary is intentionally concise
    to minimize context window usage while preserving enough information
    for sub-agents to understand the conversation history.

    Args:
        exchanges: The list of Exchange instances to summarize.

    Returns:
        A multi-line string summarizing the conversation topics and key points.
    """

    # Build a summary with one line per exchange capturing the topic.
    summary_lines: list[str] = []
    summary_lines.append(
        f"Summary of {len(exchanges)} earlier exchanges in this session:"
    )

    for index, exchange in enumerate(exchanges, start=1):
        # Truncate long queries/responses to keep the summary compact.
        query_preview: str = _truncate_text(exchange.user_query, max_length=80)
        response_preview: str = _truncate_text(exchange.system_response, max_length=120)
        summary_lines.append(
            f"  [{index}] Q: {query_preview} | A: {response_preview}"
        )

    summary_text: str = "\n".join(summary_lines)

    return summary_text


def _truncate_text(
    text: str,
    max_length: int,
) -> str:
    """
    Truncate text to a maximum length, appending ellipsis if truncated.

    Args:
        text: The text to potentially truncate.
        max_length: Maximum allowed character count before truncation.

    Returns:
        The original text if within limits, or a truncated version with "..."
        appended.
    """

    if len(text) <= max_length:
        return text

    truncated: str = text[:max_length] + "..."

    return truncated
