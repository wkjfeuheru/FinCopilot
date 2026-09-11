import asyncio
import json

import httpx
import pytest

from finharness.provider.anthropic_compat import AnthropicCompatProvider
from finharness.provider.errors import AuthError, NetworkError, RateLimitError, ServerError
from finharness.types import ModelUsage, Msg, StreamEvent, ToolUse, ToolUseDelta


def sse_response(events=None, *, lines=None):
    if lines is None:
        lines = [f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in (events or [])]
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(lines).encode())


def collect(handler, *, messages=None, tools=None, **kwargs):
    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicCompatProvider(
            base_url="https://api.anthropic.com/v1", api_key="secret", model="claude-test",
            api_version="2023-06-01", temperature=0.2, max_tokens=256, client=client,
            **kwargs,
        )
        chunks = [chunk async for chunk in provider.stream(
            system="系统提示", messages=messages or [Msg.user("问题")], tools=tools or [], usage=ModelUsage()
        )]
        await client.aclose()
        return chunks
    return asyncio.run(run())


def test_anthropic_request_conversion():
    async def handler(request):
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.headers["content-type"] == "application/json"
        body = json.loads(request.content)
        assert body["system"] == "系统提示"
        assert body["model"] == "claude-test"
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 256
        assert body["stream"] is True
        assert body["tools"] == [{"name": "get_quote", "description": "报价", "input_schema": {"type": "object"}}]
        assert body["messages"][0] == {"role": "user", "content": "查报价"}
        assert body["messages"][1] == {"role": "assistant", "content": [
            {"type": "text", "text": "思考"},
            {"type": "tool_use", "id": "call_1", "name": "get_quote", "input": {"symbol": "600519"}},
        ]}
        assert body["messages"][-1]["content"] == [{"type": "tool_result", "tool_use_id": "call_1", "content": "100"}]
        return sse_response([{"type": "message_stop"}])

    messages = [
        Msg.user("查报价"),
        Msg(role="assistant", content="思考", tool_uses=[ToolUse("call_1", "get_quote", {"symbol": "600519"})]),
        Msg(role="tool_result", content=None, tool_results=[("call_1", "100")]),
    ]
    tools = [{"type": "function", "function": {"name": "get_quote", "description": "报价", "parameters": {"type": "object"}}}]
    chunks = collect(handler, messages=messages, tools=tools)
    assert chunks[-1].event == StreamEvent.MESSAGE_END


def test_anthropic_text_tool_usage_and_single_end():
    async def handler(request):
        return sse_response([
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "你好"}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_quote", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"symbol":"600519"}'}},
            {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
            {"type": "message_delta", "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ])
    chunks = collect(handler)
    assert [c.event for c in chunks] == [StreamEvent.TEXT_DELTA, StreamEvent.TOOL_USE_DELTA, StreamEvent.TOOL_USE_DELTA, StreamEvent.MESSAGE_END]
    assert chunks[1].data == ToolUseDelta(index=1, call_id="toolu_1", name_delta="get_quote")
    assert chunks[-1].data.tool_uses == [ToolUse("toolu_1", "get_quote", {"symbol": "600519"})]
    assert chunks[-1].data.input_tokens == 3 and chunks[-1].data.output_tokens == 2


@pytest.mark.parametrize(("status", "error_type"), [(401, AuthError), (403, AuthError), (429, RateLimitError), (500, ServerError), (400, NetworkError)])
def test_anthropic_http_status_errors(status, error_type):
    async def handler(request):
        return httpx.Response(status, content=b"error")
    with pytest.raises(error_type):
        collect(handler)


@pytest.mark.parametrize(("kind", "error_type"), [("rate_limit_error", RateLimitError), ("authentication_error", AuthError), ("permission_error", AuthError), ("server_error", ServerError), ("invalid_request_error", NetworkError)])
def test_anthropic_sse_error_classification(kind, error_type):
    async def handler(request):
        return sse_response([{"type": "error", "error": {"type": kind, "message": "failure"}}])
    with pytest.raises(error_type):
        collect(handler)


def test_anthropic_malformed_and_eof():
    async def bad(request):
        return sse_response(lines=["data: {bad}\n\n"])
    with pytest.raises(NetworkError, match="invalid SSE JSON"):
        collect(bad)

    async def eof(request):
        return sse_response(lines=['data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n'])
    chunks = collect(eof)
    assert [c.event for c in chunks] == [StreamEvent.TEXT_DELTA, StreamEvent.MESSAGE_END]


def test_anthropic_timeout_maps_network_error():
    class SlowLines(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.05)
            yield b"data: {\"type\":\"message_stop\"}\n\n"
        async def aclose(self):
            return None

    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowLines())

    with pytest.raises(NetworkError, match="first response byte"):
        collect(handler, first_byte_timeout_s=0.001)


def test_anthropic_idle_timeout_maps_network_error():
    class SlowLines(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n'
            await asyncio.sleep(0.05)
            yield b'data: {"type":"message_stop"}\n\n'
        async def aclose(self):
            return None

    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowLines())

    with pytest.raises(NetworkError, match="idle timeout"):
        collect(handler, idle_timeout_s=0.001)


def test_anthropic_httpx_error_maps_network_error():
    async def handler(request):
        raise httpx.ConnectError("connection failed", request=request)

    with pytest.raises(NetworkError, match="connection failed"):
        collect(handler)


def test_anthropic_missing_usage_emits_zero_usage():
    async def handler(request):
        return sse_response([{"type": "message_stop"}])

    chunks = collect(handler)
    assert chunks[-1].data.input_tokens == 0
    assert chunks[-1].data.output_tokens == 0


@pytest.mark.parametrize("event", [
    [],
    {"type": "message_start", "message": []},
    {"type": "message_start", "message": {"usage": []}},
    {"type": "message_delta", "usage": []},
    {"type": "message_start", "message": {"usage": {"input_tokens": True}}},
    {"type": "message_delta", "usage": {"output_tokens": "2"}},
    {"type": "content_block_start", "index": 0, "content_block": []},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": 1, "name": "tool", "input": {}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "unknown"}},
    {"type": [], "message": {}},
    {"type": {"name": "message_delta"}, "usage": {}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": []}},
    {"type": "content_block_delta", "index": 0, "delta": []},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": 1}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "unknown"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": [], "text": "x"}},
])
def test_anthropic_malformed_structures_are_network_errors(event):
    async def handler(request):
        return sse_response([event])

    with pytest.raises(NetworkError):
        collect(handler)
