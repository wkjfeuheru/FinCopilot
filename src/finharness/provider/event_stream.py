"""共享的 SSE 帧解析与工具调用累积器。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from finharness.provider.errors import NetworkError
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
        buffer = self._buffers.setdefault(delta.index, _ToolUseBuffer())
        if delta.call_id:
            buffer.call_id += delta.call_id
        if delta.name_delta:
            buffer.name += delta.name_delta
        if delta.arguments_delta:
            buffer.arguments += delta.arguments_delta

    def replace_arguments(self, index: int, arguments: str = "") -> None:
        buffer = self._buffers.setdefault(index, _ToolUseBuffer())
        buffer.arguments = arguments

    def build(self) -> list[ToolUse]:
        tool_uses: list[ToolUse] = []
        for _, buffer in sorted(self._buffers.items()):
            if not buffer.call_id or not buffer.name:
                raise NetworkError("Provider returned incomplete tool call")
            try:
                arguments = json.loads(buffer.arguments or "{}")
            except json.JSONDecodeError as error:
                raise NetworkError("Provider returned invalid tool arguments") from error
            if not isinstance(arguments, dict):
                raise NetworkError("Provider returned invalid tool arguments")
            tool_uses.append(
                ToolUse(call_id=buffer.call_id, name=buffer.name, args=arguments)
            )
        return tool_uses


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
