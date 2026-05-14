"""
Account Seizure Sub-Agent module.

Implements the specialized sub-agent for answering legal questions about
account seizures (Kontopfändungen) in German banking operations. This agent
covers seizure order processing, protected amounts (Pfändungsfreigrenzen),
third-party debt orders, and priority of claims, citing ZPO and PfÜB
provisions in its responses.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from agents.base import BaseSubAgent
from models import (
    ConfidenceLevel,
    ConversationContext,
    LegalReference,
    SubAgentMetadata,
    SubAgentResponse,
)


# Knowledge base entries mapping topics to legal provisions and explanations.
# Each entry contains keywords for matching, the legal references, a template
# answer body, and the confidence level when matched.
_KNOWLEDGE_BASE: list[dict] = [
    {
        "category": "seizure_order",
        "keywords": [
            "pfändungsbeschluss",
            "zustellung",
            "pfändung",
            "seizure order",
            "pfändungs- und überweisungsbeschluss",
            "kontopfändung",
            "account seizure",
            "pfüb",
            "vollstreckung",
            "enforcement",
        ],
        "references": [
            LegalReference(law_name="ZPO", paragraph="§ 829", section="Abs. 1"),
            LegalReference(law_name="ZPO", paragraph="§ 835", section="Abs. 1"),
        ],
        "answer_de": (
            "Bei Zustellung eines Pfändungs- und Überweisungsbeschlusses (PfÜB) "
            "gemäß § 829 Abs. 1 ZPO wird das Guthaben des Schuldners auf dem "
            "betroffenen Konto mit sofortiger Wirkung gepfändet. Die Bank als "
            "Drittschuldnerin darf ab Zustellung keine Auszahlungen mehr an den "
            "Schuldner vornehmen. Die Überweisung an den Gläubiger erfolgt nach "
            "§ 835 Abs. 1 ZPO. Die Bank muss innerhalb von zwei Wochen eine "
            "Drittschuldnererklärung abgeben (§ 840 ZPO)."
        ),
        "answer_en": (
            "Upon service of a seizure and transfer order (Pfändungs- und "
            "Überweisungsbeschluss, PfÜB) pursuant to § 829(1) ZPO, the debtor's "
            "balance in the affected account is seized with immediate effect. The "
            "bank as third-party debtor must not make any further payments to the "
            "debtor from the moment of service. Transfer to the creditor follows "
            "§ 835(1) ZPO. The bank must submit a third-party debtor declaration "
            "within two weeks (§ 840 ZPO)."
        ),
        "confidence": ConfidenceLevel.HIGH,
    },
    {
        "category": "protected_amounts",
        "keywords": [
            "pfändungsfreigrenzen",
            "pfändungsfreibetrag",
            "freibetrag",
            "protected amount",
            "unpfändbar",
            "existenzminimum",
            "p-konto",
            "pfändungsschutzkonto",
            "basispfändungsschutz",
            "grundfreibetrag",
            "protection account",
        ],
        "references": [
            LegalReference(law_name="ZPO", paragraph="§ 850c"),
            LegalReference(law_name="ZPO", paragraph="§ 850k", section="Abs. 1"),
            LegalReference(law_name="ZPO", paragraph="§ 899", section="Abs. 1"),
        ],
        "answer_de": (
            "Die Pfändungsfreigrenzen sind in § 850c ZPO geregelt und werden "
            "jährlich angepasst. Bei einem Pfändungsschutzkonto (P-Konto) gemäß "
            "§ 850k Abs. 1 ZPO steht dem Schuldner ein monatlicher Grundfreibetrag "
            "automatisch zur Verfügung (§ 899 Abs. 1 ZPO). Dieser Basispfändungsschutz "
            "gilt ohne gesonderten Gerichtsbeschluss. Erhöhungsbeträge für "
            "Unterhaltspflichten können auf Antrag festgesetzt werden."
        ),
        "answer_en": (
            "Protected amounts (Pfändungsfreigrenzen) are regulated in § 850c ZPO "
            "and adjusted annually. With a garnishment protection account (P-Konto) "
            "pursuant to § 850k(1) ZPO, the debtor automatically retains a monthly "
            "base allowance (§ 899(1) ZPO). This basic protection applies without "
            "a separate court order. Increased amounts for maintenance obligations "
            "can be set upon application."
        ),
        "confidence": ConfidenceLevel.HIGH,
    },
    {
        "category": "third_party_debt_order",
        "keywords": [
            "drittschuldner",
            "drittschuldnererklärung",
            "third-party debt",
            "third party debtor",
            "auskunftspflicht",
            "erklärungspflicht",
            "disclosure obligation",
        ],
        "references": [
            LegalReference(law_name="ZPO", paragraph="§ 840", section="Abs. 1"),
            LegalReference(law_name="ZPO", paragraph="§ 840", section="Abs. 2"),
        ],
        "answer_de": (
            "Die Bank ist als Drittschuldnerin gemäß § 840 Abs. 1 ZPO verpflichtet, "
            "innerhalb von zwei Wochen nach Zustellung des Pfändungsbeschlusses eine "
            "Drittschuldnererklärung abzugeben. Darin muss sie Auskunft erteilen über: "
            "1) ob und inwieweit sie die Forderung als begründet anerkennt, "
            "2) ob andere Personen Ansprüche auf die Forderung erheben, und "
            "3) ob und wegen welcher Ansprüche die Forderung bereits gepfändet ist. "
            "Bei schuldhafter Nichtabgabe haftet die Bank dem Gläubiger auf "
            "Schadensersatz (§ 840 Abs. 2 ZPO)."
        ),
        "answer_en": (
            "As third-party debtor, the bank is obligated under § 840(1) ZPO to "
            "submit a third-party debtor declaration within two weeks of service of "
            "the seizure order. The declaration must disclose: 1) whether and to what "
            "extent the bank acknowledges the claim, 2) whether other persons assert "
            "claims to the receivable, and 3) whether and for which claims the "
            "receivable has already been seized. Culpable failure to submit the "
            "declaration exposes the bank to damages liability toward the creditor "
            "(§ 840(2) ZPO)."
        ),
        "confidence": ConfidenceLevel.HIGH,
    },
    {
        "category": "priority_of_claims",
        "keywords": [
            "rangfolge",
            "priorität",
            "priority",
            "reihenfolge",
            "mehrfachpfändung",
            "multiple seizure",
            "vorrang",
            "nachrang",
            "prioritätsprinzip",
        ],
        "references": [
            LegalReference(law_name="ZPO", paragraph="§ 804", section="Abs. 3"),
            LegalReference(law_name="ZPO", paragraph="§ 829", section="Abs. 1"),
        ],
        "answer_de": (
            "Bei mehreren Pfändungen desselben Kontos gilt das Prioritätsprinzip "
            "gemäß § 804 Abs. 3 ZPO: Die Rangfolge richtet sich nach dem Zeitpunkt "
            "der Zustellung des jeweiligen Pfändungsbeschlusses an die Bank. Der "
            "zuerst zugestellte Pfändungsbeschluss hat Vorrang. Die Bank muss die "
            "Reihenfolge der Zustellungen dokumentieren und bei der Verteilung "
            "des Guthabens beachten. Ein später zugestellter Pfändungsbeschluss "
            "wird erst bedient, wenn der vorrangige Gläubiger vollständig "
            "befriedigt ist."
        ),
        "answer_en": (
            "When multiple seizures affect the same account, the priority principle "
            "under § 804(3) ZPO applies: priority is determined by the time of "
            "service of each seizure order on the bank. The seizure order served "
            "first takes precedence. The bank must document the sequence of service "
            "and observe it when distributing the balance. A later-served seizure "
            "order is only satisfied once the prior creditor has been fully paid."
        ),
        "confidence": ConfidenceLevel.HIGH,
    },
]


class AccountSeizureAgent(BaseSubAgent):
    """
    Sub-agent specialized in German account seizure law (Kontopfändungsrecht).

    Handles queries about seizure order processing, protected amounts
    (Pfändungsfreigrenzen), third-party debt orders, and priority of claims.
    Cites ZPO and PfÜB provisions in all substantive responses.

    The agent matches incoming queries against a structured knowledge base
    of legal provisions. When a match is found, it returns the applicable
    legal answer with citations and HIGH confidence. When no provision
    matches but the query is within scope, it returns LOW confidence with
    a limitation note. When the query is entirely outside the agent's
    domain, it flags the response as out-of-scope.
    """

    def get_metadata(self) -> SubAgentMetadata:
        """
        Return metadata describing the account seizure agent's capabilities.

        Returns:
            SubAgentMetadata with domain_id "account_seizure", a description
            of covered topics, and the list of supported query categories.
        """
        return SubAgentMetadata(
            domain_id="account_seizure",
            description=(
                "Handles legal questions about account seizures "
                "(Kontopfändungen) in German banking operations, including "
                "seizure order processing, protected amounts "
                "(Pfändungsfreigrenzen), third-party debt orders, and "
                "priority of claims."
            ),
            supported_categories=[
                "seizure_order",
                "protected_amounts",
                "third_party_debt_order",
                "priority_of_claims",
            ],
        )

    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """
        Process a legal query about account seizures.

        Matches the query against the knowledge base using keyword matching.
        Returns a structured response with legal references and confidence
        assessment.

        Args:
            query: The user's natural-language legal question about account
                seizures, already classified as belonging to this domain.
            context: Conversation history for the current session.

        Returns:
            SubAgentResponse with the answer, legal references, confidence
            level, and scope/limitation flags.
        """
        query_lower: str = query.lower()

        # Determine language from query content (simple heuristic).
        is_german: bool = _detect_german(query_lower)

        # Try to match against knowledge base entries.
        best_match: dict | None = None
        best_score: int = 0

        for entry in _KNOWLEDGE_BASE:
            score: int = sum(
                1 for keyword in entry["keywords"]
                if keyword in query_lower
            )
            if score > best_score:
                best_score = score
                best_match = entry

        # If we found a matching provision, return a substantive answer.
        if best_match is not None and best_score > 0:
            answer_body: str = (
                best_match["answer_de"] if is_german else best_match["answer_en"]
            )
            return SubAgentResponse(
                domain_id="account_seizure",
                answer_body=answer_body,
                references=list(best_match["references"]),
                confidence=best_match["confidence"],
                is_out_of_scope=False,
                limitation_note=None,
            )

        # Check if the query is at least within the general domain of
        # account seizures (but no specific provision matched).
        if _is_within_domain(query_lower):
            limitation_note: str = (
                "Zu dieser spezifischen Fragestellung konnte keine eindeutige "
                "gesetzliche Regelung im Bereich der Kontopfändung identifiziert "
                "werden. Eine individuelle rechtliche Beratung wird empfohlen."
                if is_german
                else "No specific legal provision in the area of account seizures "
                "could be identified for this particular question. Individual "
                "legal consultation is recommended."
            )
            return SubAgentResponse(
                domain_id="account_seizure",
                answer_body=limitation_note,
                references=[],
                confidence=ConfidenceLevel.LOW,
                is_out_of_scope=False,
                limitation_note=limitation_note,
            )

        # Query is outside this agent's scope entirely.
        return SubAgentResponse(
            domain_id="account_seizure",
            answer_body="",
            references=[],
            confidence=ConfidenceLevel.LOW,
            is_out_of_scope=True,
            limitation_note=None,
        )


def _detect_german(query_lower: str) -> bool:
    """
    Simple heuristic to detect whether a query is in German.

    Checks for common German words and characters. This is a lightweight
    detection for response language selection — the actual language validation
    happens at the executor layer.

    Args:
        query_lower: Lowercased query string.

    Returns:
        True if the query appears to be in German, False otherwise.
    """
    german_indicators: list[str] = [
        "pfändung",
        "konto",
        "bank",
        "gläubiger",
        "schuldner",
        "forderung",
        "beschluss",
        "gericht",
        "betrag",
        "freibetrag",
        "zustellung",
        "wie",
        "was",
        "wann",
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
    Determine whether a query is within the account seizure domain.

    Even if no specific provision matches, the query may still be about
    account seizures in general. This function checks for broad domain
    indicators.

    Args:
        query_lower: Lowercased query string.

    Returns:
        True if the query appears to be about account seizures.
    """
    domain_indicators: list[str] = [
        "pfändung",
        "seizure",
        "garnishment",
        "kontopfändung",
        "pfüb",
        "vollstreckung",
        "enforcement",
        "zwangsvollstreckung",
        "gläubiger",
        "creditor",
        "schuldner",
        "debtor",
        "forderung",
        "claim",
        "p-konto",
        "freibetrag",
        "protected",
        "drittschuldner",
    ]
    return any(indicator in query_lower for indicator in domain_indicators)
