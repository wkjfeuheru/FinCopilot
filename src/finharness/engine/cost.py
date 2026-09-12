"""Session statistics for usage, retries and tool execution time."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SessionStatsSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    retry_count: int = 0
    tool_calls: int = 0
    tool_duration_ms: int = 0
    per_tool: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    # Sub-agent usage, keyed by focus (e.g. "risk"). A sub-agent's tokens are
    # also folded into input_tokens/output_tokens above, so the total stays
    # complete while this dimension shows where the spend went.
    per_agent: Mapping[str, Mapping[str, int]] = field(default_factory=dict)


class SessionStats:
    """Mutable session-local counters; snapshots are detached read-only copies."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.retry_count = 0
        self.tool_calls = 0
        self._clock = clock
        self._per_tool: dict[str, dict[str, int]] = {}
        self._per_agent: dict[str, dict[str, int]] = {}

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record_agent_usage(self, name: str, input_tokens: int, output_tokens: int) -> None:
        """Attribute usage to a sub-agent focus as a breakdown, not a second sum.

        The totals are incremented separately via ``add_usage``; keeping the two
        apart means a snapshot can report both "how much in total" and "how much
        of it was the risk reviewer" without double-counting.
        """
        entry = self._per_agent.setdefault(
            name, {"input_tokens": 0, "output_tokens": 0, "runs": 0}
        )
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["runs"] += 1

    def add_retry(self) -> None:
        self.retry_count += 1

    def record_tool_request(self, name: str) -> None:
        """Count every tool request, including unknown or rejected ones."""
        self.tool_calls += 1
        entry = self._per_tool.setdefault(name, {"count": 0, "duration_ms": 0})
        entry["count"] += 1

    def now(self) -> float:
        return self._clock()

    def record_tool_duration(self, name: str, started_at: float) -> int:
        """Accumulate wall time for a call that actually entered tool.run()."""
        duration_ms = round((self._clock() - started_at) * 1000)
        entry = self._per_tool.setdefault(name, {"count": 0, "duration_ms": 0})
        entry["duration_ms"] += duration_ms
        return duration_ms

    def snapshot(self) -> SessionStatsSnapshot:
        return SessionStatsSnapshot(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            retry_count=self.retry_count,
            tool_calls=self.tool_calls,
            tool_duration_ms=sum(entry["duration_ms"] for entry in self._per_tool.values()),
            per_tool=MappingProxyType(
                {name: MappingProxyType(dict(entry)) for name, entry in self._per_tool.items()}
            ),
            per_agent=MappingProxyType(
                {name: MappingProxyType(dict(entry)) for name, entry in self._per_agent.items()}
            ),
        )
