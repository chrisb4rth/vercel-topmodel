"""
Query Classifier module.

Provides the QueryClassifier class responsible for classifying incoming user
queries into one or more legal sub-domains based on the metadata of registered
sub-agents. The classifier uses deterministic keyword matching — comparing
normalized query tokens against each sub-agent's description and supported
categories — to produce a ClassificationResult.

The classifier is intentionally simple and LLM-free: it tokenizes the query,
checks for overlap with each domain's keywords (extracted from description and
supported_categories), and returns all matching domain_ids. This approach is
transparent, fast, and testable via property-based tests.

Key behaviors:
    - Returns ALL matching domain_ids for cross-domain queries (Requirement 1.2).
    - Returns EMPTY domain_ids when no domain matches, allowing the supervisor
      to respond with a clarification listing available sub-domains (Requirement 1.3).
    - Every returned domain_id is guaranteed to exist in the available_domains
      list (Property 3 from design).
    - Confidence is derived from the proportion of keyword hits relative to
      the total keywords available for the best-matching domain.
    - Language detection is performed via simple heuristic (common German
      stopwords presence) and defaults to English when uncertain.

Requirements: 1.1, 1.2, 1.3
"""

import logging
import re

from models import ClassificationResult, Language, SubAgentMetadata

# Module-level logger for classification diagnostics.
logger: logging.Logger = logging.getLogger(__name__)

# Common German stopwords and function words used for language detection.
# If a query contains several of these, it is likely German.
_GERMAN_INDICATORS: set[str] = {
    "der", "die", "das", "ein", "eine", "und", "oder", "ist", "sind",
    "wird", "werden", "hat", "haben", "für", "auf", "mit", "von", "zu",
    "bei", "nach", "über", "unter", "wie", "was", "wer", "wenn", "kann",
    "nicht", "auch", "noch", "nur", "aber", "als", "dem", "den", "des",
    "im", "am", "vom", "zum", "zur", "ich", "wir", "sie", "er", "es",
    "mein", "sein", "ihr", "kein", "keine", "welche", "welcher", "welches",
    "dieser", "diese", "dieses", "jeder", "jede", "jedes", "müssen", "sollen",
    "können", "dürfen", "möchten", "wollen", "bereits", "jedoch", "daher",
    "deshalb", "trotzdem", "obwohl", "während", "bevor", "nachdem",
}

# Minimum number of German indicator words required to classify as German.
_GERMAN_DETECTION_THRESHOLD: int = 2


class QueryClassifier:
    """
    Deterministic query classifier for legal sub-domain routing.

    The classifier operates by extracting keywords from each registered
    sub-agent's metadata (description + supported_categories) and comparing
    them against the normalized tokens of an incoming query. Domains with
    at least one keyword hit are considered matches.

    This design ensures:
        - Extensibility: new domains are automatically included in classification
          when their metadata contains descriptive keywords.
        - Transparency: classification decisions are traceable to specific
          keyword overlaps, aiding debugging and auditing.
        - Safety: every returned domain_id is guaranteed to exist in the
          provided available_domains list (Property 3).

    The classifier does NOT call an LLM — it is a fast, deterministic component
    suitable for the hot path of every incoming request.
    """

    def __init__(self) -> None:
        """
        Initialize the QueryClassifier.

        No configuration is required at construction time. All domain metadata
        is provided per-call via the `available_domains` parameter, keeping the
        classifier stateless and easy to test.
        """

        return None

    async def classify_query(
        self,
        query: str,
        available_domains: list[SubAgentMetadata],
    ) -> ClassificationResult:
        """
        Classify a query into one or more legal sub-domains.

        Performs keyword-based matching of the query against each domain's
        metadata (description and supported_categories). Returns all domains
        that have at least one keyword overlap with the query tokens.

        The classification pipeline:
            1. Normalize the query to lowercase and tokenize.
            2. For each domain, extract keywords from description and categories.
            3. Compute keyword overlap (intersection of query tokens and domain keywords).
            4. Collect all domains with non-zero overlap as matches.
            5. Compute confidence from the best match's hit ratio.
            6. Detect query language via German stopword heuristic.

        Args:
            query: The user's natural-language legal question. Expected to be
                pre-validated (1–2000 characters, supported language).
            available_domains: Metadata for all registered sub-agents, provided
                by the registry. The classifier only returns domain_ids that
                appear in this list.

        Returns:
            A ClassificationResult containing:
                - domain_ids: List of matched domain identifiers (may be empty).
                - confidence: Float 0.0–1.0 based on keyword hit quality.
                - language: Detected language of the query (German or English).
        """

        # Normalize and tokenize the query for case-insensitive matching.
        query_tokens: set[str] = self._tokenize(query)

        # Detect the language of the query using stopword heuristics.
        detected_language: Language = self._detect_language(query_tokens)

        # If no domains are available, return empty classification immediately.
        if not available_domains:
            logger.warning(
                "No available domains provided to classifier. "
                "Returning empty classification result."
            )
            empty_result: ClassificationResult = ClassificationResult(
                domain_ids=[],
                confidence=0.0,
                language=detected_language,
            )
            return empty_result

        # Score each domain by counting keyword hits against the query tokens.
        domain_scores: list[tuple[str, int, int]] = []
        for domain in available_domains:
            domain_keywords: set[str] = self._extract_domain_keywords(domain)
            hit_count: int = len(query_tokens & domain_keywords)

            if hit_count > 0:
                domain_scores.append((domain.domain_id, hit_count, len(domain_keywords)))

        # Collect all domain_ids that had at least one keyword hit.
        matched_domain_ids: list[str] = [
            domain_id
            for domain_id, _hits, _total in domain_scores
        ]

        # Compute confidence from the best-matching domain's hit ratio.
        confidence: float = self._compute_confidence(domain_scores)

        logger.info(
            "Classified query into %d domain(s): %s (confidence=%.2f, language=%s)",
            len(matched_domain_ids),
            matched_domain_ids,
            confidence,
            detected_language.value,
        )

        classification_result: ClassificationResult = ClassificationResult(
            domain_ids=matched_domain_ids,
            confidence=confidence,
            language=detected_language,
        )
        return classification_result

    def _tokenize(
        self,
        text: str,
    ) -> set[str]:
        """
        Normalize and tokenize a text string into a set of lowercase words.

        Splits on non-alphanumeric characters (preserving umlauts and other
        Unicode letters) and filters out very short tokens (length < 2) that
        are unlikely to be meaningful keywords.

        Args:
            text: Raw text string to tokenize.

        Returns:
            A set of unique lowercase tokens extracted from the text.
        """

        # Lowercase the entire text for case-insensitive comparison.
        lowered: str = text.lower()

        # Split on non-word characters (Unicode-aware), keeping meaningful tokens.
        raw_tokens: list[str] = re.findall(r"[\w]+", lowered, re.UNICODE)

        # Filter out single-character tokens that add noise without signal.
        meaningful_tokens: set[str] = {
            token
            for token in raw_tokens
            if len(token) >= 2
        }

        return meaningful_tokens

    def _extract_domain_keywords(
        self,
        metadata: SubAgentMetadata,
    ) -> set[str]:
        """
        Extract searchable keywords from a sub-agent's metadata.

        Combines the domain's description and supported_categories into a single
        keyword set. Category names are split on underscores to produce individual
        words (e.g., "seizure_order" → {"seizure", "order"}).

        Args:
            metadata: The sub-agent metadata containing description and categories.

        Returns:
            A set of lowercase keyword tokens representing the domain's coverage.
        """

        # Tokenize the description to extract meaningful words.
        description_tokens: set[str] = self._tokenize(metadata.description)

        # Tokenize each category, splitting on underscores before tokenizing.
        category_tokens: set[str] = set()
        for category in metadata.supported_categories:
            # Replace underscores with spaces so tokenizer splits them.
            expanded_category: str = category.replace("_", " ")
            category_tokens.update(self._tokenize(expanded_category))

        # Combine both sources into a unified keyword set.
        all_keywords: set[str] = description_tokens | category_tokens

        return all_keywords

    def _compute_confidence(
        self,
        domain_scores: list[tuple[str, int, int]],
    ) -> float:
        """
        Compute a confidence score based on the best domain's keyword hit ratio.

        Confidence is the ratio of keyword hits to total domain keywords for the
        domain with the highest hit count. This reflects how strongly the query
        aligns with the best-matching domain. If no domains matched, confidence
        is 0.0.

        The score is clamped to [0.0, 1.0] for safety.

        Args:
            domain_scores: List of tuples (domain_id, hit_count, total_keywords)
                for domains that had at least one keyword hit.

        Returns:
            A float between 0.0 and 1.0 representing classification confidence.
        """

        # No matches means zero confidence.
        if not domain_scores:
            return 0.0

        # Find the domain with the most keyword hits.
        best_score: tuple[str, int, int] = max(
            domain_scores,
            key=lambda entry: entry[1],
        )
        _best_domain_id, best_hits, best_total = best_score

        # Avoid division by zero (should not happen with valid metadata).
        if best_total == 0:
            return 0.0

        # Ratio of hits to total keywords, clamped to [0.0, 1.0].
        raw_confidence: float = best_hits / best_total
        clamped_confidence: float = min(1.0, max(0.0, raw_confidence))

        return clamped_confidence

    def _detect_language(
        self,
        query_tokens: set[str],
    ) -> Language:
        """
        Detect the language of a query using German stopword heuristics.

        Counts how many tokens in the query match known German function words.
        If the count meets or exceeds the detection threshold, the query is
        classified as German. Otherwise, it defaults to English.

        This is a simple heuristic suitable for the two-language system. It
        does not attempt full NLP-based language detection — the validation
        layer has already confirmed the query is in a supported language.

        Args:
            query_tokens: Pre-tokenized set of lowercase words from the query.

        Returns:
            Language.GERMAN if sufficient German indicators are found,
            Language.ENGLISH otherwise.
        """

        # Count how many query tokens are known German function words.
        german_indicator_count: int = len(query_tokens & _GERMAN_INDICATORS)

        # Apply threshold to decide language.
        if german_indicator_count >= _GERMAN_DETECTION_THRESHOLD:
            return Language.GERMAN

        return Language.ENGLISH
