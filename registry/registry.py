"""
Sub-Agent Registry module.

Provides the SubAgentRegistry class responsible for storing, discovering, and
retrieving legal domain sub-agents. The registry is the central mechanism through
which the supervisor discovers available sub-agents at startup — enabling new
legal domains to be added without modifying supervisor or classifier logic.

Sub-agents register themselves by providing their metadata (domain_id, description,
supported_categories). The registry enforces uniqueness of domain_id to prevent
accidental overwrites and provides lookup methods for both targeted retrieval
(by domain_id list) and full introspection (all metadata).

Requirements: 8.1, 8.2, 8.3
"""

import logging

from agents.base import BaseSubAgent
from models import SubAgentMetadata

# Module-level logger for registry operations and error reporting.
logger: logging.Logger = logging.getLogger(__name__)


class SubAgentRegistry:
    """
    Registry for discovering and managing legal domain sub-agents.

    The registry stores sub-agents keyed by their unique domain_id, enabling
    the supervisor to:
        1. Discover all available sub-agents at startup via `get_all_metadata()`.
        2. Retrieve specific sub-agents for classified domains via
           `get_agents_for_domains()`.
        3. Detect configuration errors (duplicate registrations, empty registry).

    The registry is populated during application startup before any queries are
    processed. Once populated, it is treated as read-only during request handling.

    Attributes:
        _agents: Internal dictionary mapping domain_id strings to their
            corresponding BaseSubAgent instances. Keyed by domain_id to
            ensure O(1) lookup during dispatch.
    """

    def __init__(self) -> None:
        """
        Initialize an empty sub-agent registry.

        The registry starts empty and is populated via successive `register()`
        calls during application startup.
        """

        # Internal storage: domain_id → BaseSubAgent instance.
        self._agents: dict[str, BaseSubAgent] = {}

        return None

    def register(
        self,
        agent: BaseSubAgent,
    ) -> None:
        """
        Register a sub-agent by its domain identifier.

        Extracts the agent's metadata to determine its domain_id, then stores
        the agent in the internal dictionary. Raises ValueError if a sub-agent
        with the same domain_id is already registered — this prevents silent
        overwrites that could cause routing confusion.

        Args:
            agent: A concrete implementation of BaseSubAgent to register.
                Must have a unique domain_id in its metadata.

        Raises:
            ValueError: If a sub-agent with the same domain_id is already
                registered in this registry instance.

        Returns:
            None
        """

        # Extract metadata to obtain the domain_id for keying.
        metadata: SubAgentMetadata = agent.get_metadata()
        domain_id: str = metadata.domain_id

        # Guard against duplicate registrations to prevent silent overwrites.
        if domain_id in self._agents:
            raise ValueError(
                f"Sub-agent with domain_id '{domain_id}' is already registered. "
                f"Each domain_id must be unique within the registry."
            )

        # Store the agent keyed by its domain_id for O(1) lookup.
        self._agents[domain_id] = agent
        logger.info(
            "Registered sub-agent for domain '%s': %s",
            domain_id,
            metadata.description,
        )

        return None

    def get_agents_for_domains(
        self,
        domain_ids: list[str],
    ) -> list[BaseSubAgent]:
        """
        Retrieve sub-agents matching the given domain identifiers.

        Looks up each requested domain_id in the registry and returns the
        corresponding sub-agents. Domain IDs that do not match any registered
        agent are silently skipped — the caller (dispatcher) is responsible for
        handling missing domains if needed.

        If the registry is empty when this method is called, an error is logged
        per Requirement 8.3 (system not ready).

        Args:
            domain_ids: List of domain identifier strings to look up. These
                typically come from a ClassificationResult after query classification.

        Returns:
            A list of BaseSubAgent instances matching the requested domain_ids.
            Returns an empty list if no domain_ids match or if the registry is empty.
        """

        # Log an error if the registry has no agents — indicates misconfiguration.
        if not self._agents:
            logger.error(
                "Registry is empty: no sub-agents are available. "
                "The system cannot process queries without registered sub-agents."
            )
            return []

        # Collect matching agents, skipping unrecognized domain_ids.
        matched_agents: list[BaseSubAgent] = [
            self._agents[domain_id]
            for domain_id in domain_ids
            if domain_id in self._agents
        ]

        return matched_agents

    def get_all_metadata(self) -> list[SubAgentMetadata]:
        """
        Return metadata for all registered sub-agents.

        Provides the supervisor's classifier with a complete view of available
        sub-domains so it can determine which agents are relevant for a given
        query. Each metadata entry includes the domain_id, description, and
        supported_categories.

        If the registry is empty when this method is called, an error is logged
        per Requirement 8.3 (system not ready).

        Returns:
            A list of SubAgentMetadata instances, one per registered sub-agent.
            Returns an empty list if no sub-agents are registered.
        """

        # Log an error if the registry has no agents — indicates misconfiguration.
        if not self._agents:
            logger.error(
                "Registry is empty: no sub-agents are available. "
                "Cannot provide metadata for classification."
            )
            return []

        # Collect metadata from all registered agents in insertion order.
        all_metadata: list[SubAgentMetadata] = [
            agent.get_metadata()
            for agent in self._agents.values()
        ]

        return all_metadata
