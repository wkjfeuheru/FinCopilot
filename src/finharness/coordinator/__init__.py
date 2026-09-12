"""Multi-agent support (docs 03.10).

Only one focus ships: risk. The macro focus in the docs needs a ``web_search``
tool that does not exist, and the generic ``spawn_agent`` surface was cut with
it — a coordinator with one caller does not need a framework.
"""

from finharness.coordinator.reviewer import (
    RISK_FOCUS,
    RISK_SKILL,
    Coordinator,
    SubAgentResult,
)

__all__ = ["Coordinator", "SubAgentResult", "RISK_FOCUS", "RISK_SKILL"]
