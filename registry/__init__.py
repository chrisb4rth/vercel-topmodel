"""
Registry package.

Provides the SubAgentRegistry for discovering and managing sub-agents.
The supervisor discovers available sub-agents through this registry at startup,
enabling new legal domains without supervisor code changes.
"""

from registry.registry import SubAgentRegistry

__all__: list[str] = ["SubAgentRegistry"]
