"""Hook 链：围绕工具执行的调用前/调用后拦截（docs 03.7.3）。"""

from __future__ import annotations

from abc import ABC
from typing import Any

from finharness.types import ToolResult


class BaseHook(ABC):
    """一个调用前/调用后拦截器。``pre`` 可以阻断；``post`` 只做观察。"""

    async def pre(self, tool, args: dict, *, turn: int = 0) -> bool:
        """返回 False 以阻断执行。"""
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
        """观察一次结果。绝不能向循环抛出异常。"""
        return None


class HookChain:
    """按顺序运行 hook；某个阻断性的 ``pre`` 会使链条短路。"""

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
            except Exception:  # noqa: BLE001 - 审计绝不能破坏一次回合
                continue
