"""
Insolvency Sub-Agent module.

Implements the specialized sub-agent for answering legal questions about
insolvency proceedings (Insolvenzverfahren) in German banking operations.
This agent covers account blocking upon insolvency filing (§ 89 InsO),
insolvency administrator rights (§ 80 InsO), payment prohibitions (§ 82 InsO),
and estate segregation (§ 35 InsO), citing InsO provisions in its responses.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from agents.base import BaseSubAgent
from models import (
    ConfidenceLevel,
    ConversationContext,
    LegalReference,
    SubAgentMetadata,
    SubAgentResponse,
)


# Knowledge base entries mapping insolvency topics to legal provisions.
# Each entry contains keywords for matching, the legal references, a template
# answer body, and the confidence level when matched.
_KNOWLEDGE_BASE: list[dict] = [
    {
        "category": "account_blocking",
        "keywords": [
            "kontosperre",
            "sperre",
            "sperrung",
            "gesperrt",
            "verfügungsverbot",
            "account blocking",
            "block",
            "blocked",
            "eröffnung",
            "insolvenzantrag",
            "filing",
            "vollstreckungsverbot",
            "einzelzwangsvollstreckung",
            "§ 89",
            "konto",
        ],
        "references": [
            LegalReference(law_name="InsO", paragraph="§ 89", section="Abs. 1"),
            LegalReference(law_name="InsO", paragraph="§ 21", section="Abs. 2"),
        ],
        "answer_de": (
            "Mit Eröffnung des Insolvenzverfahrens tritt gemäß § 89 Abs. 1 InsO "
            "ein Vollstreckungsverbot ein: Zwangsvollstreckungen für einzelne "
            "Insolvenzgläubiger sind weder in die Insolvenzmasse noch in das "
            "sonstige Vermögen des Schuldners zulässig. Die Bank muss daher "
            "sämtliche Kontoverfügungen des Schuldners unterbinden, soweit sie "
            "nicht vom Insolvenzverwalter genehmigt sind. Bereits im "
            "Eröffnungsverfahren kann das Gericht nach § 21 Abs. 2 InsO "
            "Sicherungsmaßnahmen anordnen, einschließlich eines allgemeinen "
            "Verfügungsverbots."
        ),
        "answer_en": (
            "Upon opening of insolvency proceedings, an enforcement prohibition "
            "takes effect pursuant to § 89(1) InsO: individual enforcement actions "
            "by insolvency creditors are not permitted against either the insolvency "
            "estate or the debtor's other assets. The bank must therefore block all "
            "account dispositions by the debtor unless authorized by the insolvency "
            "administrator. Even during the preliminary proceedings, the court may "
            "order protective measures under § 21(2) InsO, including a general "
            "prohibition on dispositions."
        ),
        "confidence": ConfidenceLevel.HIGH,
    },
    {
        "category": "administrator_rights",
        "keywords": [
            "insolvenzverwalter",
            "verwalter",
            "verwaltungs- und verfügungsbefugnis",
            "verfügungsbefugnis",
            "befugnis",
            "administrator",
            "insolvency administrator",
            "trustee",
            "verwaltungsrecht",
            "§ 80",
        ],
        "references": [
            LegalReference(law_name="InsO", paragraph="§ 80", section="Abs. 1"),
            LegalReference(law_name="InsO", paragraph="§ 81", section="Abs. 1"),
        ],
        "answer_de": (
            "Mit Eröffnung des Insolvenzverfahrens geht gemäß § 80 Abs. 1 InsO "
            "das Recht des Schuldners, das zur Insolvenzmasse gehörende Vermögen "
            "zu verwalten und über es zu verfügen, auf den Insolvenzverwalter über. "
            "Die Bank muss daher ausschließlich Weisungen des Insolvenzverwalters "
            "bezüglich der Konten des Schuldners befolgen. Verfügungen des "
            "Schuldners über Gegenstände der Insolvenzmasse nach Verfahrenseröffnung "
            "sind gemäß § 81 Abs. 1 InsO unwirksam."
        ),
        "answer_en": (
            "Upon opening of insolvency proceedings, the debtor's right to manage "
            "and dispose of assets belonging to the insolvency estate passes to the "
            "insolvency administrator pursuant to § 80(1) InsO. The bank must "
            "therefore exclusively follow instructions from the insolvency "
            "administrator regarding the debtor's accounts. Dispositions by the "
            "debtor over objects of the insolvency estate after opening of "
            "proceedings are void under § 81(1) InsO."
        ),
        "confidence": ConfidenceLevel.HIGH,
    },
    {
        "category": "payment_prohibitions",
        "keywords": [
            "zahlungsverbot",
            "leistungsverbot",
            "zahlung",
            "payment prohibition",
            "payment ban",
            "payment",
            "schuldbefreiung",
            "befreiende wirkung",
            "§ 82",
        ],
        "references": [
            LegalReference(law_name="InsO", paragraph="§ 82", section="Abs. 1"),
        ],
        "answer_de": (
            "Gemäß § 82 Abs. 1 InsO wird der Schuldner eines Insolvenzgläubigers "
            "von seiner Verbindlichkeit befreit, wenn er nach Eröffnung des "
            "Insolvenzverfahrens an den Schuldner (Insolvenzschuldner) leistet, "
            "sofern er die Eröffnung zum Zeitpunkt der Leistung nicht kannte. "
            "Für die Bank bedeutet dies: Zahlungen an den Insolvenzschuldner nach "
            "Verfahrenseröffnung haben nur dann befreiende Wirkung, wenn die Bank "
            "keine Kenntnis von der Insolvenzeröffnung hatte. Ab Kenntnis darf die "
            "Bank nur noch an den Insolvenzverwalter leisten."
        ),
        "answer_en": (
            "Under § 82(1) InsO, a debtor of an insolvency creditor is discharged "
            "from their obligation if they perform to the debtor (insolvency debtor) "
            "after opening of insolvency proceedings, provided they were unaware of "
            "the opening at the time of performance. For the bank this means: "
            "payments to the insolvency debtor after opening of proceedings only "
            "have discharging effect if the bank had no knowledge of the insolvency "
            "opening. Once aware, the bank may only perform to the insolvency "
            "administrator."
        ),
        "confidence": ConfidenceLevel.HIGH,
    },
    {
        "category": "estate_segregation",
        "keywords": [
            "insolvenzmasse",
            "masse",
            "massebestandteil",
            "vermögen",
            "insolvency estate",
            "estate",
            "segregation",
            "absonderung",
            "aussonderung",
            "neuerwerb",
            "§ 35",
        ],
        "references": [
            LegalReference(law_name="InsO", paragraph="§ 35", section="Abs. 1"),
            LegalReference(law_name="InsO", paragraph="§ 36", section="Abs. 1"),
        ],
        "answer_de": (
            "Die Insolvenzmasse umfasst gemäß § 35 Abs. 1 InsO das gesamte "
            "Vermögen, das dem Schuldner zur Zeit der Eröffnung des Verfahrens "
            "gehört und das er während des Verfahrens erlangt (Neuerwerb). "
            "Für die Bank bedeutet dies, dass sämtliche Kontoguthaben des "
            "Schuldners zum Zeitpunkt der Eröffnung sowie alle nachfolgenden "
            "Eingänge zur Insolvenzmasse gehören. Nicht zur Masse gehören "
            "gemäß § 36 Abs. 1 InsO unpfändbare Gegenstände."
        ),
        "answer_en": (
            "The insolvency estate comprises, pursuant to § 35(1) InsO, all assets "
            "belonging to the debtor at the time of opening of proceedings and "
            "assets acquired during the proceedings (new acquisitions). For the "
            "bank this means that all account balances of the debtor at the time "
            "of opening as well as all subsequent incoming payments belong to the "
            "insolvency estate. Assets exempt from seizure under § 36(1) InsO do "
            "not form part of the estate."
        ),
        "confidence": ConfidenceLevel.HIGH,
    },
]


class InsolvencyAgent(BaseSubAgent):
    """
    Sub-agent specialized in German insolvency law (Insolvenzrecht).

    Handles queries about account blocking upon insolvency filing (§ 89 InsO),
    insolvency administrator rights (§ 80 InsO), payment prohibitions
    (§ 82 InsO), and estate segregation (§ 35 InsO). Cites InsO provisions
    in all substantive responses.

    The agent matches incoming queries against a structured knowledge base
    of InsO provisions. When a match is found, it returns the applicable
    legal answer with citations and HIGH confidence. When no provision
    matches but the query is within scope, it returns LOW confidence with
    a limitation note. When the query is entirely outside the agent's
    domain, it flags the response as out-of-scope.
    """

    def get_metadata(self) -> SubAgentMetadata:
        """
        Return metadata describing the insolvency agent's capabilities.

        Returns:
            SubAgentMetadata with domain_id "insolvency", a description
            of covered topics, and the list of supported query categories.
        """
        return SubAgentMetadata(
            domain_id="insolvency",
            description=(
                "Handles legal questions about insolvency proceedings "
                "(Insolvenzverfahren) in German banking operations, including "
                "account blocking upon insolvency filing, insolvency "
                "administrator rights, payment prohibitions, and estate "
                "segregation."
            ),
            supported_categories=[
                "account_blocking",
                "administrator_rights",
                "payment_prohibitions",
                "estate_segregation",
            ],
        )

    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """
        Process a legal query about insolvency proceedings.

        Matches the query against the knowledge base using keyword matching.
        Returns a structured response with legal references and confidence
        assessment.

        Args:
            query: The user's natural-language legal question about insolvency
                proceedings, already classified as belonging to this domain.
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
                domain_id="insolvency",
                answer_body=answer_body,
                references=list(best_match["references"]),
                confidence=best_match["confidence"],
                is_out_of_scope=False,
                limitation_note=None,
            )

        # Check if the query is at least within the general domain of
        # insolvency (but no specific provision matched).
        if _is_within_domain(query_lower):
            limitation_note: str = (
                "Zu dieser spezifischen Fragestellung konnte keine eindeutige "
                "Regelung in der Insolvenzordnung identifiziert werden. Eine "
                "individuelle rechtliche Beratung wird empfohlen."
                if is_german
                else "No specific provision in the Insolvency Statute (InsO) "
                "could be identified for this particular question. Individual "
                "legal consultation is recommended."
            )
            return SubAgentResponse(
                domain_id="insolvency",
                answer_body=limitation_note,
                references=[],
                confidence=ConfidenceLevel.LOW,
                is_out_of_scope=False,
                limitation_note=limitation_note,
            )

        # Query is outside this agent's scope entirely.
        return SubAgentResponse(
            domain_id="insolvency",
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
        "insolvenz",
        "verwalter",
        "verfahren",
        "masse",
        "gläubiger",
        "schuldner",
        "eröffnung",
        "gericht",
        "zahlung",
        "konto",
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
    Determine whether a query is within the insolvency domain.

    Even if no specific provision matches, the query may still be about
    insolvency proceedings in general. This function checks for broad
    domain indicators.

    Args:
        query_lower: Lowercased query string.

    Returns:
        True if the query appears to be about insolvency proceedings.
    """
    domain_indicators: list[str] = [
        "insolvenz",
        "insolvency",
        "insolvent",
        "verwalter",
        "administrator",
        "trustee",
        "masse",
        "estate",
        "eröffnung",
        "opening",
        "gläubigerversammlung",
        "creditors meeting",
        "restschuldbefreiung",
        "discharge",
        "insolvenzplan",
        "insolvency plan",
        "absonderung",
        "separation",
        "aussonderung",
        "segregation",
    ]
    return any(indicator in query_lower for indicator in domain_indicators)
