"""FinHarness 各层共享的数据契约。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


@dataclass(slots=True)
class ToolUse:
    call_id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolUseDelta:
    index: int
    call_id: str | None = None
    name_delta: str = ""
    arguments_delta: str = ""


@dataclass(slots=True)
class Msg:
    role: str
    content: str | None
    tool_uses: list[ToolUse] = field(default_factory=list)
    tool_results: list[tuple[str, str]] = field(default_factory=list)
    # 传输/UI 元数据随对话记录一起持久化，但绝不发送给
    # 模型 provider。它让恢复的回答能够还原其可观测的运行。
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, content: str) -> Msg:
        return cls(role="user", content=content)


# 会话研究状态块作为**尾部 user 消息**随请求发送（docs 3.3）：它是本请求的一个视图，
# 从不是用户输入、也从不持久化。``role`` 无法把它与真实提问区分开，因此用元数据标记，
# 使「最后一条用户消息」这一读法（脚本化 provider、测试替身）不会把状态块当成问题。
STATE_VIEW_META = "state_view"


class StreamEvent(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_USE_DELTA = "tool_use_delta"
    MESSAGE_END = "message_end"
    ERROR = "error"


@dataclass(slots=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_uses: list[ToolUse] = field(default_factory=list)
    # 前缀缓存计费。provider 对缓存输入的计费远低于新输入，
    # 因此这一拆分才能让"这次优化是否奏效"可回答，
    # 而不只是一句主张。未上报时保持 0；当 provider 不区分时
    # cache_miss_tokens 保持 0（input_tokens 仍是总量）。
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


@dataclass(slots=True)
class StreamChunk:
    event: StreamEvent
    data: Any = None


@dataclass(slots=True)
class EngineEvent:
    kind: str
    data: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    content: str
    ok: bool = True
    error: str | None = None
    attachments: list[str] = field(default_factory=list)
    # 该结果据此渲染的原始载荷；loop 将它们登记为
    # 引用，并用分配的 id 填充 ``citations``。
    sources: list[Any] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    # 工具声明的可观测性载荷，从 ``RawData.metadata`` 转发给
    # hook 读取。不持久化到任何地方，也绝不发送给 provider。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObservedCall:
    """一个轮次内单次工具调用的观测（Thought/Action/Observation 中的 O）。

    对于从未执行的调用，``ok`` 为 False——被拒绝的路径、激活晚了一轮的
    惰性工具、loop-guard 的拒绝——因为这些未执行恰恰是安全/行为评估
    需要看到的。``preview`` 是渲染后观测的定长开头，
    使磁盘上的 trace 保持精简。
    """

    call_id: str
    name: str
    ok: bool
    error: str | None = None
    preview: str = ""
    duration_ms: int = 0


@dataclass(slots=True)
class RoundTrace:
    """一次 loop 迭代：模型的文本、它发起的调用及其结果。

    每一轮都会记录（不只是成功的轮次），使一次运行可以作为轨迹
    重放，并与预期路径比较。``thought`` 即使在轮次最终是工具调用时
    也保留文本——loop 会从回答中丢弃该草稿，但它是模型陈述的推理，
    也正是轨迹记录的意义所在。``answer`` 仅在终止轮次设置。
    """

    turn: int
    thought: str = ""
    actions: list[ToolUse] = field(default_factory=list)
    observations: list[ObservedCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    # 从该轮 provider 调用到首个文本增量的墙钟时间，再到
    # 流结束；按轮记录，使缓慢的轮次可归因。
    llm_first_ms: int = 0
    llm_ms: int = 0
    answer: str = ""


@dataclass(slots=True)
class AgentTurnOutcome:
    answer: str
    succeeded: bool = True
    usage: ModelUsage = field(default_factory=ModelUsage)
    error: str | None = None
    reason: str | None = None
    tool_calls: int = 0
    retry_count: int = 0
    tool_duration_ms: int = 0
    citations: list[str] = field(default_factory=list)
    # 完整的 Thought/Action/Observation 记录，最早的轮次在前。对
    # 早于它出现的调用方为空；纯增量，忽略它的代码不会出问题。
    trace: list[RoundTrace] = field(default_factory=list)
    rounds: int = 0


class OutputSink(Protocol):
    """引擎事件的输出汇点；服务层借此接收流式事件并推送给客户端。"""

    async def emit(self, event: EngineEvent) -> None: ...


class StopSignal:
    """一次挂起请求的协作式中断信号（docs 03.3）。

    放在共享契约里，因为它的两端分属不同层：服务层创建并置位它，引擎在等待点
    检查它。引擎不依赖服务层，因此这个原语不能住在任何一侧。

    它是一个事件而非布尔量：引擎只需 ``requested`` 一次判断即可，无需轮询也
    无需 await，因此检查点本身不会成为新的挂起点，也不会改变没有停止请求时
    的执行时序。
    """

    __slots__ = ("_event", "reason")

    def __init__(self, reason: str = "user_stopped") -> None:
        self._event = asyncio.Event()
        self.reason = reason

    def request(self, reason: str | None = None) -> None:
        """置位；重复置位幂等（用户可能连点两次停止）。"""
        if reason:
            self.reason = reason
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()
