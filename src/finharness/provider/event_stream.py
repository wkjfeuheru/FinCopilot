"""共享的 SSE 帧解析与工具调用累积器。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from finharness.provider.errors import (
    MalformedStreamError,
    NetworkError,
    OutputTruncatedError,
    ToolArgumentsError,
)
from finharness.types import ToolUse, ToolUseDelta


@dataclass(slots=True)
class _ToolUseBuffer:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


class ToolUseAccumulator:
    """按流中的索引累积工具调用片段。"""

    def __init__(self) -> None:
        self._buffers: dict[int, _ToolUseBuffer] = {}

    def add(self, delta: ToolUseDelta) -> None:
        """累积一个工具调用增量，按 index 归入对应缓冲区。"""
        buffer = self._buffers.setdefault(delta.index, _ToolUseBuffer())
        if delta.call_id:
            buffer.call_id += delta.call_id
        if delta.name_delta:
            buffer.name += delta.name_delta
        if delta.arguments_delta:
            buffer.arguments += delta.arguments_delta

    def replace_arguments(self, index: int, arguments: str = "") -> None:
        """重置指定 index 已累积的参数（在 JSON 片段开始前调用）。"""
        buffer = self._buffers.setdefault(index, _ToolUseBuffer())
        buffer.arguments = arguments

    def build(self) -> list[ToolUse]:
        """将所有缓冲区构建为校验后的 ToolUse 列表；字段缺失或 JSON 非法时抛出。"""
        tool_uses: list[ToolUse] = []
        for _, buffer in sorted(self._buffers.items()):
            if not buffer.call_id or not buffer.name:
                raise ToolArgumentsError("Provider returned incomplete tool call")
            try:
                arguments = json.loads(buffer.arguments or "{}")
            except json.JSONDecodeError as error:
                raise ToolArgumentsError("Provider returned invalid tool arguments") from error
            if not isinstance(arguments, dict):
                raise ToolArgumentsError("Provider returned invalid tool arguments")
            tool_uses.append(
                ToolUse(call_id=buffer.call_id, name=buffer.name, args=arguments)
            )
        return tool_uses


def finalize_tool_uses(
    accumulator: ToolUseAccumulator, *, finish_reason: str | None = None
) -> list[ToolUse]:
    """构建工具调用；并用 ``finish_reason`` 区分"被截断"与"坏 JSON"。

    参数解析失败有两类成因，处理方式不同：模型吐了坏 JSON（概率性，重试可解），
    或输出触到 ``max_tokens`` 被服务端截断（预算不足，重试大概率仍截断，但值得
    一次机会并应报出 ``OutputTruncatedError`` 让运维看到该调大预算）。

    没有这个区分时，截断会被贴上 ``NetworkError`` 标签，看起来像网络故障，
    真正的"输出不够长"信号就此丢失。
    """
    try:
        return accumulator.build()
    except MalformedStreamError as error:
        if finish_reason == "length":
            raise OutputTruncatedError(
                "Provider output was truncated at max_tokens; "
                "tool arguments are incomplete. Raise model.max_tokens."
            ) from error
        raise


async def iter_sse_data(
    lines: AsyncIterator[str],
    *,
    first_byte_timeout_s: float,
    idle_timeout_s: float,
) -> AsyncIterator[str]:
    """从 SSE 行迭代器中产生每个事件的聚合 ``data`` 内容。"""
    data_lines: list[str] = []
    first_read = True

    while True:
        try:
            timeout_s = first_byte_timeout_s if first_read else idle_timeout_s
            async with asyncio.timeout(timeout_s):
                line = await anext(lines)
        except StopAsyncIteration:
            if data_lines:
                yield "\n".join(data_lines)
            return
        except TimeoutError as error:
            message = (
                "Provider timed out waiting for first response byte"
                if first_read
                else "Provider stream idle timeout"
            )
            raise NetworkError(message, retryable=first_read) from error

        first_read = False
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
