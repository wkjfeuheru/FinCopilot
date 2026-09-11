import asyncio

import pytest

from finharness.provider.errors import NetworkError
from finharness.provider.event_stream import ToolUseAccumulator, iter_sse_data
from finharness.types import ToolUseDelta


def test_tool_use_accumulator_joins_fragmented_arguments():
    """删除任一参数片段应使最终工具参数不完整。"""
    accumulator = ToolUseAccumulator()
    accumulator.add(ToolUseDelta(index=1, call_id="call-1", name_delta="quote"))
    accumulator.add(ToolUseDelta(index=1, arguments_delta='{"symbol":'))
    accumulator.add(ToolUseDelta(index=1, arguments_delta='"600519"}'))

    assert accumulator.build()[0].args == {"symbol": "600519"}


def test_tool_use_accumulator_rejects_invalid_arguments():
    """若不将参数解析为 JSON 对象，提供方的错误参数会被静默接受。"""
    accumulator = ToolUseAccumulator()
    accumulator.add(
        ToolUseDelta(
            index=0,
            call_id="call-1",
            name_delta="quote",
            arguments_delta="not-json",
        )
    )

    with pytest.raises(NetworkError, match="^Provider returned invalid tool arguments$"):
        accumulator.build()


def test_tool_use_accumulator_replaces_initial_arguments():
    """若替换未清空旧缓冲区，最终参数会包含过期内容。"""
    accumulator = ToolUseAccumulator()
    accumulator.add(
        ToolUseDelta(
            index=0,
            call_id="call-1",
            name_delta="quote",
            arguments_delta='{"symbol":"old"}',
        )
    )
    accumulator.replace_arguments(0, '{"symbol":"new"}')

    assert accumulator.build()[0].args == {"symbol": "new"}


def test_tool_use_accumulator_uses_empty_object_for_empty_arguments():
    accumulator = ToolUseAccumulator()
    accumulator.add(ToolUseDelta(index=0, call_id="call-1", name_delta="quote"))

    assert accumulator.build()[0].args == {}


def test_tool_use_accumulator_sorts_by_index():
    accumulator = ToolUseAccumulator()
    accumulator.add(ToolUseDelta(index=2, call_id="call-2", name_delta="second"))
    accumulator.add(ToolUseDelta(index=1, call_id="call-1", name_delta="first"))

    assert [tool.call_id for tool in accumulator.build()] == ["call-1", "call-2"]


@pytest.mark.parametrize(
    "delta",
    [
        ToolUseDelta(index=0, name_delta="quote"),
        ToolUseDelta(index=0, call_id="call-1"),
    ],
)
def test_tool_use_accumulator_requires_call_id_and_name(delta):
    accumulator = ToolUseAccumulator()
    accumulator.add(delta)

    with pytest.raises(NetworkError, match="incomplete tool call"):
        accumulator.build()


def test_tool_use_accumulator_rejects_non_object_json():
    accumulator = ToolUseAccumulator()
    accumulator.add(
        ToolUseDelta(index=0, call_id="call-1", name_delta="quote", arguments_delta="[]")
    )

    with pytest.raises(NetworkError, match="^Provider returned invalid tool arguments$"):
        accumulator.build()


def test_iter_sse_data_joins_multiline_data_and_ignores_comments():
    """若事件帧未按 SSE 规则聚合，拆分的 data 内容无法被正确消费。"""
    async def lines():
        yield ": keepalive\n"
        yield "event: message\n"
        yield "data: first\n"
        yield "data: second\n"
        yield "\n"

    async def collect():
        return [
            data
            async for data in iter_sse_data(
                lines(), first_byte_timeout_s=0.1, idle_timeout_s=0.1
            )
        ]

    assert asyncio.run(collect()) == ["first\nsecond"]


def test_iter_sse_data_times_out_before_first_response_byte():
    """缺失首字节超时会让尚未响应的请求永久等待。"""
    async def lines():
        await asyncio.sleep(0.05)
        yield "data: late\n"

    async def collect():
        return [
            data
            async for data in iter_sse_data(
                lines(), first_byte_timeout_s=0.001, idle_timeout_s=0.1
            )
        ]

    with pytest.raises(NetworkError, match="first response byte"):
        asyncio.run(collect())


def test_iter_sse_data_emits_unfinished_frame_at_eof_and_ignores_unknown_fields():
    async def lines():
        yield "id: 42\n"
        yield "retry: 1000\n"
        yield "data: unfinished\n"

    async def collect():
        return [
            data
            async for data in iter_sse_data(
                lines(), first_byte_timeout_s=0.1, idle_timeout_s=0.1
            )
        ]

    assert asyncio.run(collect()) == ["unfinished"]


def test_iter_sse_data_times_out_when_stream_goes_idle():
    async def lines():
        yield "data: first\n"
        await asyncio.sleep(0.05)
        yield "data: late\n"

    async def collect():
        return [
            data
            async for data in iter_sse_data(
                lines(), first_byte_timeout_s=0.1, idle_timeout_s=0.001
            )
        ]

    with pytest.raises(NetworkError, match="stream idle"):
        asyncio.run(collect())
