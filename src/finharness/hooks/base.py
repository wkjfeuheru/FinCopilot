"""Hook chain: pre/post interception around tool execution (docs 03.7.3)."""

from __future__ import annotations

from abc import ABC
from typing import Any

from finharness.types import ToolResult


class BaseHook(ABC):
    """A pre/post interceptor. ``pre`` may block; ``post`` only observes."""

    async def pre(self, tool, args: dict, *, turn: int = 0) -> bool:
        """Return False to block execution."""
        return True

    async def post(
        self,
        tool,
        args: dict,
        result: ToolResult,
        *,
        action: str,
        verdict: str,
        duration_ms: float = 0.0,
        citations: list[str] | None = None,
        turn: int = 0,
        endpoint: str = "",
        rows: int = 0,
        cols: int = 0,
    ) -> None:
        """Observe an outcome. Must never raise into the loop."""
        return None


class HookChain:
    """Runs hooks in order; a blocking ``pre`` short-circuits the chain."""

    def __init__(self, hooks: list[BaseHook] | None = None) -> None:
        self.hooks: list[BaseHook] = list(hooks or [])

    def add(self, hook: BaseHook) -> None:
        self.hooks.append(hook)

    async def pre(self, tool, args: dict, *, turn: int = 0) -> bool:
        for hook in self.hooks:
            if not await hook.pre(tool, args, turn=turn):
                return False
        return True

    async def post(self, tool, args: dict, result: ToolResult, **kwargs: Any) -> None:
        for hook in self.hooks:
            try:
                await hook.post(tool, args, result, **kwargs)
            except Exception:  # noqa: BLE001 - audit must never break a turn
                continue
