import asyncio

import pytest

from finharness.provider.errors import (
    MalformedStreamError,
    NetworkError,
    OutputTruncatedError,
    ToolArgumentsError,
)
from finharness.provider.event_stream import (
    ToolUseAccumulator,
    finalize_tool_uses,
    iter_sse_data,
)
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


def test_finalize_classifies_truncated_tool_arguments_as_truncation():
    """finish_reason=length 时参数残缺，应报"输出被截断"而非笼统的网络错误。

    这是运维能否看到"max_tokens 不够用"这一信号的关键：没有这层区分，截断会
    被贴上 NetworkError 标签，看起来像网络抖动。
    """
    accumulator = ToolUseAccumulator()
    accumulator.add(
        ToolUseDelta(
            index=0,
            call_id="call-1",
            name_delta="write_report",
            arguments_delta='{"topic": "宁德时代", "sections": [{"heading": "营收',
        )
    )

    with pytest.raises(OutputTruncatedError, match="max_tokens"):
        finalize_tool_uses(accumulator, finish_reason="length")


def test_finalize_keeps_bad_arguments_separate_from_truncation():
    """非 length 的坏 JSON 报坏参数——模型吐错了，而不是预算不足。"""
    accumulator = ToolUseAccumulator()
    accumulator.add(
        ToolUseDelta(
            index=0, call_id="call-1", name_delta="get_quote", arguments_delta="not-json"
        )
    )

    with pytest.raises(ToolArgumentsError) as error:
        finalize_tool_uses(accumulator, finish_reason="tool_calls")
    assert not isinstance(error.value, OutputTruncatedError)


def test_finalize_passes_through_valid_arguments():
    accumulator = ToolUseAccumulator()
    accumulator.add(
        ToolUseDelta(
            index=0, call_id="call-1", name_delta="get_quote", arguments_delta='{"symbol":"600519"}'
        )
    )

    assert finalize_tool_uses(accumulator, finish_reason="tool_calls")[0].args == {
        "symbol": "600519"
    }


def test_malformed_stream_errors_are_retryable_and_void_output():
    """这一类失败发生在流结束之后，重放前必须先作废已流出的内容。"""
    error = OutputTruncatedError("truncated")

    assert isinstance(error, MalformedStreamError)
    assert isinstance(error, NetworkError)
    assert error.retryable is True
    assert error.voids_output is True

    # 普通传输层 NetworkError 不得作废输出：它发生在任何 chunk 之前。
    assert NetworkError("connection refused").voids_output is False

