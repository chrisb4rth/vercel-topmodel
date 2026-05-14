"""
Base sub-agent interface module.

Defines the abstract base class that all legal domain sub-agents must implement.
This module establishes the contract between the supervisor/registry layer and
individual sub-agents, ensuring uniform query handling and metadata reporting
across all legal domains.

The BaseSubAgent class enforces two core operations:
    1. handle_query — Process a classified legal query with conversation context
       and return a structured response including legal references and confidence.
    2. get_metadata — Report the agent's capabilities so the registry and
       classifier can discover and route queries appropriately.

By coding to this interface, new legal domains (e.g., data protection, AML)
can be added without modifying the supervisor or registry logic — they simply
implement BaseSubAgent and register themselves.

Requirements: 8.4, 8.5
"""

from abc import ABC, abstractmethod

from models import ConversationContext, SubAgentMetadata, SubAgentResponse


class BaseSubAgent(ABC):
    """
    Abstract base class for all legal domain sub-agents.

    Every sub-agent in the system must extend this class and provide concrete
    implementations of `handle_query` and `get_metadata`. This guarantees that:

    - The registry can introspect any agent's capabilities via `get_metadata`.
    - The dispatcher can invoke any agent uniformly via `handle_query`.
    - Response formatting (references, confidence, scope flags) is consistent
      across all domains.

    Subclasses are responsible for:
        - Maintaining their own knowledge base or retrieval mechanism.
        - Assigning appropriate confidence levels based on provision match quality.
        - Setting `is_out_of_scope=True` when a query falls outside their domain.
        - Populating `limitation_note` when no matching provision is found.
    """

    @abstractmethod
    async def handle_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> SubAgentResponse:
        """
        Process a classified legal query and return a structured response.

        This method receives a natural-language query that has already been
        classified as relevant to this agent's domain. The agent should:
            1. Analyze the query against its knowledge base.
            2. Identify applicable legal provisions.
            3. Formulate an answer with citations and confidence assessment.

        Args:
            query: The user's natural-language legal question, already classified
                as belonging to this agent's domain. Between 1 and 2000 characters.
            context: The conversation history for the current session, enabling
                the agent to resolve references to prior exchanges (e.g., "the
                amount mentioned earlier").

        Returns:
            A SubAgentResponse containing the answer body, legal references,
            confidence level, and scope/limitation flags.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> SubAgentMetadata:
        """
        Return metadata describing this agent's capabilities.

        The registry calls this method during agent registration to catalog
        the agent's domain, description, and supported query categories. The
        classifier then uses this metadata to determine which agents should
        handle a given query.

        Returns:
            A SubAgentMetadata instance with:
                - domain_id: Unique identifier for this legal sub-domain.
                - description: Human-readable summary of covered topics.
                - supported_categories: List of query category strings this
                  agent can handle (used by the classifier for routing).
        """
        ...
