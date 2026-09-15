"""L1 工作记忆：原始对话记录及其 token 核算的持有者。

Docs 03.6.4 (2)：这是原始消息列表的唯一写入者。``AgentLoop`` 负责编排；
窗口维护（我现在有多满、是否必须压缩、用摘要替换某个区间）放在这里，这样
loop 就不会多出第二项职责。
"""

from __future__ import annotations

from dataclasses import dataclass

from finharness.config.settings import Settings
from finharness.context.tokens import TokenCounter
from finharness.types import Msg

# 当未给出显式保留数量时，最近多少轮会在压缩后原样保留
# （通常由预算决定；这里是下限）。
KEEP_RECENT_ROUNDS = 2
WORKING_WINDOW_FLOOR = KEEP_RECENT_ROUNDS


@dataclass(frozen=True, slots=True)
class WindowUsage:
    """两个不得混为一谈的不同数字。

    ``window_tokens`` 是下一次请求会有多满 —— 压缩所缩减的那个数值。
    ``used_tokens`` 是会话的累计消耗，压缩绝不能改变它，因为它是计费数字。
    """

    window_tokens: int
    used_tokens: int
    exact: bool


class WorkingMemory:
    """原始对话记录加上增量式 token 核算。"""

    def __init__(self, *, ctx=None, settings: Settings, counter: TokenCounter | None = None) -> None:
        self.settings = settings
        self.ctx = ctx
        # 计数器的词表缓存是项目级的，而非按数据目录划分。
        self.counter = counter or TokenCounter()
        self.raw: list[Msg] = []
        self.used_tokens = 0
        self._exact = True
        # 目前已从窗口折叠出去的消息；也即下一条记录行的序号，
        # 用于把摘要分段关联到某段消息区间。
        self.discarded = 0
        # 本轮追加消息的可选暂存处，使 loop 能一次性批量持久化，
        # 而不必逐条写入。
        self.pending: list[Msg] = []

    # -- 写入（唯一的变更点） --------------------------------------------------
    def append(self, message: Msg, *, track: bool = True) -> None:
        """追加一条消息；对回放的历史使用 ``track=False``。

        从存储中加载回来的消息不得缓冲为 pending，否则下一次持久化会把它们
        再写一遍，造成序号重复。
        """
        self.raw.append(message)
        if track:
            self.pending.append(message)
        self._account([message])

    def append_user(self, text: str) -> None:
        self.append(Msg.user(text))

    def append_assistant(self, message: Msg) -> None:
        self.append(message)

    def append_tool_result(self, call_id: str, content: str) -> None:
        self.append(Msg(role="tool_result", content=None, tool_results=[(call_id, content)]))

    # -- 读取 ------------------------------------------------------------------
    def snapshot(self) -> list[Msg]:
        """供 provider 请求使用的只读副本。"""
        return list(self.raw)

    def _account(self, messages: list[Msg]) -> None:
        """把若干消息的 token 累加进会话累计消耗，并跟踪计数是否精确。"""
        for message in messages:
            for text in _message_texts(message):
                counted = self.counter.count(text)
                self.used_tokens += counted.tokens
                self._exact = self._exact and counted.exact

    def request_tokens(self, *, system: str, tools: list[dict], extra_text: str = "") -> int:
        """下一次请求的估算大小（压缩所针对的数值）。

        实测而非推断：provider 会忽略交给它的 ``usage`` 对象，并在事后报告
        自己的总数，因此在发送 *之前* 了解窗口有多满的唯一方式就是在此计数。

        ``extra_text`` 承载随请求发送但不属于 ``raw`` 的内容 —— 尤其是研究状态
        块，它在每次迭代中追加到待发送消息上（docs 3.3）。遗漏它会低估窗口大小，
        让请求在压缩察觉之前就已溢出。
        """
        total = self.counter.count(system).tokens
        # _message_texts 对每条消息都返回一个生成器，因此先展平再计数。
        for message in self.raw:
            total += self.counter.count_many(_message_texts(message))
        if extra_text:
            total += self.counter.count(extra_text).tokens
        for schema in tools:
            function = schema.get("function", {})
            total += self.counter.count(str(function.get("description", ""))).tokens
            total += self.counter.count(str(function.get("parameters", ""))).tokens
        return total

    def usage(self, *, system: str, tools: list[dict], extra_text: str = "") -> WindowUsage:
        return WindowUsage(
            window_tokens=self.request_tokens(
                system=system, tools=tools, extra_text=extra_text
            ),
            used_tokens=self.used_tokens,
            exact=self._exact,
        )

    def over_budget(self, *, system: str, tools: list[dict], extra_text: str = "") -> bool:
        window = self.settings.context.context_window_tokens
        if window <= 0:
            return False
        threshold = window * self.settings.context.compaction_ratio
        return (
            self.request_tokens(system=system, tools=tools, extra_text=extra_text)
            >= threshold
        )

    # -- 窗口维护 --------------------------------------------------------------
    def squash(self, *, keep_rounds: int | None = None) -> tuple[int, int]:
        """丢弃除最近若干轮之外的内容，返回 ``(removed, discarded)``。

        ``removed`` 统计被折叠掉的消息数，``discarded`` 只统计工具结果数，
        后者是调用方所报告的。最近轮次按 token 预算选取，并带一个轮数下限
        （docs 03.6.4）：轮次大小相差一个数量级，因此固定的轮数要么浪费预算、
        要么超出预算，但低于下限又会砍掉模型当前正在处理的那一轮。

        此处不插入摘要消息。更早的历史存在于摘要层，并通过系统 prompt 注入，
        因此不会被伪装成用户说过的话 —— 也不会在下次被再次摘要。
        """
        keep = self._rounds_to_keep() if keep_rounds is None else keep_rounds
        boundary = _recent_boundary(self.raw, keep)
        if boundary <= 0:
            return (0, 0)
        discarded = sum(len(message.tool_results) for message in self.raw[:boundary])
        self.raw = self.raw[boundary:]
        self.discarded += boundary
        return (boundary, discarded)

    def _rounds_to_keep(self) -> int:
        """有多少最近轮次能放进预算，且绝不低于下限。"""
        floor = self.settings.context.min_recent_rounds
        if not self.raw:
            return floor
        budget = self.settings.context.context_window_tokens * (
            1.0 - self.settings.context.summary_budget_ratio
        )
        assistant_positions = [
            index for index, message in enumerate(self.raw) if message.role == "assistant"
        ]
        kept = 0
        used = 0
        # 从最新的一轮向前回溯，持续累加，直到再也放不下为止。
        for position in reversed(assistant_positions):
            chunk = self.raw[position:]
            chunk_tokens = sum(
                self.counter.count(text).tokens
                for message in chunk
                for text in _message_texts(message)
            )
            if kept >= floor and used + chunk_tokens > budget:
                break
            used += chunk_tokens
            kept += 1
        return max(kept, floor)

    def round_count(self) -> int:
        """以原样保留的轮次数；压缩器用它来界定自己的处理范围。"""
        return sum(1 for message in self.raw if message.role == "assistant")

    def window_floor(self) -> int:
        """当前窗口中最早一条消息的序号。"""
        return self.discarded


def _recent_boundary(messages: list[Msg], keep_rounds: int) -> int:
    """最近 ``keep_rounds`` 次模型交互开始处的索引。

    一个“轮次”是一次模型往返：一个 assistant 帧加上它的工具结果。
    有两点后果很重要：

    * 若按 *user* 消息计数，在长时间研究任务（一个问题、多次交互）中会找不到
      任何可折叠的内容 —— 而这正是压缩存在的场景。
    * 切割点必须落在 assistant 帧上，绝不能落在 assistant 帧与其工具结果之间：
      一个被保留下来的 tool_result，若其 call_id 从未被某条 assistant 消息声明过，
      会被 OpenAI 兼容的 API 拒绝。
    """
    if keep_rounds <= 0:
        return len(messages)
    assistant_positions = [
        index for index, message in enumerate(messages) if message.role == "assistant"
    ]
    if len(assistant_positions) <= keep_rounds:
        return 0
    return assistant_positions[-keep_rounds]


def _message_texts(message: Msg):
    """生成一条消息中所有计入 token 的文本片段（内容、工具调用、工具结果）。"""
    if message.content:
        yield message.content
    for tool_use in message.tool_uses:
        yield f"{tool_use.name}{tool_use.args}"
    for call_id, raw in message.tool_results:
        yield f"{call_id}{raw}"
