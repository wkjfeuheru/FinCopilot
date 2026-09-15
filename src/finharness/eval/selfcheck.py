"""用于 ``--offline`` 自检的确定性 provider（docs 03.13）。

离线模式的存在是为了证明评测工具可正常工作，而非评判模型，因此该 provider
依据问题文本以及是否已有工具结果，回放少量固定形态：一次工具轮次、一次拒答
以及一个最终答案。这足以演练全部四个评分器——包括当权限门禁拦截调用时产生
的真实拒答观测——且无需任何网络或 API 成本。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from finharness.provider.base import Provider
from finharness.types import ModelUsage, Msg, StreamChunk, StreamEvent, ToolUse


class SelfCheckProvider(Provider):
    """由内容触发的脚本化 provider；见模块 docstring。"""

    def __init__(self) -> None:
        self._counter = 0

    async def stream(
        self, *, system: str, messages: list[Msg], tools: list[dict], usage: ModelUsage
    ) -> AsyncIterator[StreamChunk]:
        question = _last_user_text(messages)
        has_tool_result = any(message.role == "tool_result" for message in messages)

        if "REFUSE" in question:
            yield StreamChunk(StreamEvent.TEXT_DELTA, "我只覆盖 A 股，无法提供该标的的报价。")
            yield _end(input_tokens=20, output_tokens=10)
            return

        if "BLOCKED:" in question and not has_tool_result:
            name = question.split("BLOCKED:", 1)[1].strip().split()[0]
            self._counter += 1
            yield _end(
                input_tokens=20,
                output_tokens=5,
                tool_uses=[ToolUse(f"c{self._counter}", name, {"path": "x", "content": "y"})],
            )
            return

        if "TOOL:" in question and not has_tool_result:
            name = question.split("TOOL:", 1)[1].strip().split()[0]
            self._counter += 1
            yield _end(
                input_tokens=20,
                output_tokens=5,
                tool_uses=[ToolUse(f"c{self._counter}", name, {"symbol": "600519"})],
            )
            return

        yield StreamChunk(StreamEvent.TEXT_DELTA, "（离线自检应答）已收到该问题。")
        yield _end(input_tokens=20, output_tokens=8)


def _end(
    *, input_tokens: int, output_tokens: int, tool_uses: list[ToolUse] | None = None
) -> StreamChunk:
    """构造一个标记消息结束的流式块，携带用量与可选工具调用。"""
    return StreamChunk(
        StreamEvent.MESSAGE_END,
        ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_uses=list(tool_uses or []),
        ),
    )


def _last_user_text(messages: list[Msg]) -> str:
    """返回最后一条非空的用户消息内容。"""
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


__all__ = ["SelfCheckProvider"]
