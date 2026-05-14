"""
Agents package.

Contains the base sub-agent interface and all specialized legal domain
sub-agents (account seizure, insolvency). Each sub-agent implements the
BaseSubAgent abstract class for registry compatibility.
"""

from agents.base import BaseSubAgent
from agents.account_seizure import AccountSeizureAgent
from agents.insolvency import InsolvencyAgent

__all__ = [
    "BaseSubAgent",
    "AccountSeizureAgent",
    "InsolvencyAgent",
]
