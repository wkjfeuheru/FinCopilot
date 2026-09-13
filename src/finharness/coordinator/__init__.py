"""Multi-agent support (docs 03.10).

Only one focus ships: risk. A macro focus is deliberately not implemented — the
web search tool it would need now exists, but a macro review produces narrative
judgement that cannot be checked the way "does this number match the data" can,
so the isolation buys no verifiable correctness. The generic ``spawn_agent``
surface was cut too: a coordinator with one caller does not need a framework.
"""

from finharness.coordinator.reviewer import (
    RISK_FOCUS,
    RISK_SKILL,
    Coordinator,
    SubAgentResult,
)

__all__ = ["Coordinator", "SubAgentResult", "RISK_FOCUS", "RISK_SKILL"]
