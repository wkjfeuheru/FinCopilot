"""会话统计：usage、重试与工具执行耗时。"""

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
    # sub-agent 的 usage，按 focus 归类（例如 "risk"）。sub-agent 的 token
    # 同时也会计入上方的 input_tokens/output_tokens，因此总量保持完整，
    # 而这个维度展示开销的去向。
    per_agent: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    # input_tokens 的 prefix-cache 拆分。对于不报告该拆分的 provider，
    # cache_miss_tokens 保持为 0，因此只有两者中至少一个非零时，hit_ratio
    # 才有意义。
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


class SessionStats:
    """会话内可变计数器；快照是分离的只读副本。"""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.retry_count = 0
        self.tool_calls = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self._clock = clock
        self._per_tool: dict[str, dict[str, int]] = {}
        self._per_agent: dict[str, dict[str, int]] = {}

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_hit_tokens += cache_hit_tokens
        self.cache_miss_tokens += cache_miss_tokens

    def cache_hit_ratio(self) -> float:
        """从 prefix cache 命中提供的、所报告可缓存输入所占的比例。

        当 provider 未报告拆分时返回 0.0，这与真正的 0% 命中率不同，但却是
        诚实的答案：没有任何报告。
        """
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return (self.cache_hit_tokens / total) if total else 0.0

    def record_agent_usage(self, name: str, input_tokens: int, output_tokens: int) -> None:
        """将 usage 归属到某个 sub-agent focus，作为明细拆分而非重复求和。

        总量通过 ``add_usage`` 单独递增；将二者分开意味着快照既能报告
        "总共多少"，也能报告"其中多少来自风险评审者"，且不会重复计数。
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
        """统计每一个 tool 请求，包括未知或被拒绝的请求。"""
        self.tool_calls += 1
        entry = self._per_tool.setdefault(name, {"count": 0, "duration_ms": 0})
        entry["count"] += 1

    def now(self) -> float:
        return self._clock()

    def record_tool_duration(self, name: str, started_at: float) -> int:
        """累加真正进入了 tool.run() 的调用的墙钟耗时。"""
        duration_ms = round((self._clock() - started_at) * 1000)
        entry = self._per_tool.setdefault(name, {"count": 0, "duration_ms": 0})
        entry["duration_ms"] += duration_ms
        return duration_ms

    def snapshot(self) -> SessionStatsSnapshot:
        """汇总当前计数器，生成一份分离的只读快照。"""
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
            cache_hit_tokens=self.cache_hit_tokens,
            cache_miss_tokens=self.cache_miss_tokens,
        )
