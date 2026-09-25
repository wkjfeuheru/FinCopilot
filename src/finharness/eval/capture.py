"""单次评测运行的事件采集与交互通道替身。

引擎循环本就会发出轨迹所需的全部事件；本模块收集这些事件，并按用例声明的
策略应答运行可能抛出的两种交互提示（``ask_user`` 与写操作确认），从而无需
人工即可端到端驱动一个用例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from finharness.permissions.gate import ConfirmationSpec
from finharness.types import EngineEvent


@dataclass(slots=True)
class Interaction:
    """一次交互往返，记录下来供评分使用。"""

    kind: str
    prompt: str
    options: list[str]
    response: str | None


class RecordingSink:
    """按顺序收集引擎循环的事件（进程内的 SSE 接口）。"""

    def __init__(self) -> None:
        self.events: list[EngineEvent] = []

    async def emit(self, event: EngineEvent) -> None:
        self.events.append(event)


class InteractionChannel:
    """依据策略应答 ask_user 与写操作确认提示。

    策略与用例 schema 一致：
      none    — 无通道：写操作确认被拒，ask_user 不作答；
      confirm — 批准写操作确认；
      deny    — 拒绝写操作确认；
      answer  — ask_user 以 ``answer`` 作答。

    Implements ``InteractivePort.prompt`` so AgentLoop FSM confirmation can
    use the same channel object passed as ``loop.interactive``.

    Does not emit ``interaction_resolved``: that edge is owned by the loop
    after ``ConfirmationResolved`` is persisted.
    """

    def __init__(self, policy: str = "answer", answer: str = "综合") -> None:
        self.policy = policy
        self.answer = answer
        self.log: list[Interaction] = []

    def reset(self, policy: str, answer: str) -> None:
        self.policy = policy
        self.answer = answer
        self.log = []

    async def prompt(self, spec: ConfirmationSpec) -> str | None:
        """InteractivePort adapter used by the agent FSM confirmation phase."""
        kind = "question" if getattr(spec, "kind", "") == "question" else "confirm"
        return await self.ask(
            kind,
            getattr(spec, "prompt", ""),
            list(getattr(spec, "options", ()) or []),
            multi_select=bool(getattr(spec, "multi_select", False)),
        )

    async def ask(
        self, kind: str, prompt: str, options: list[str], *, multi_select: bool = False
    ) -> str | None:
        """引擎循环的 ``interactive`` 回调（也是权限门禁的确认来源）。"""
        del multi_select
        if kind == "confirm":
            approved = self.policy == "confirm"
            response: str | None = "y" if approved else "n"
        elif self.policy == "answer":
            response = self.answer
        else:
            response = None
        self.log.append(
            Interaction(kind=kind, prompt=prompt, options=list(options or []), response=response)
        )
        return response

    async def confirm(self, name: str, args: dict) -> bool:
        """权限门禁的 ``confirm`` 回调：返回写操作是否获批。"""
        del args
        return await self.ask("confirm", name, []) == "y"


@dataclass(slots=True)
class CapturedTurn:
    """单轮产生的全部内容：事件、交互与结果。"""

    index: int
    user: str
    outcome: Any
    events: list[EngineEvent] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def answer(self) -> str:
        return str(getattr(self.outcome, "answer", "") or "")

    @property
    def trace(self) -> list[Any]:
        return list(getattr(self.outcome, "trace", []) or [])


__all__ = [
    "CapturedTurn",
    "Interaction",
    "InteractionChannel",
    "RecordingSink",
]
