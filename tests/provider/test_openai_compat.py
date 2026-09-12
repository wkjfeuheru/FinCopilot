import asyncio
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.provider.errors import AuthError, NetworkError, RateLimitError, ServerError
from finharness.types import ModelUsage, Msg, StreamEvent, ToolUse, ToolUseDelta


def sse_response(events: list[dict] | None = None, *, lines: list[str] | None = None) -> httpx.Response:
    if lines is None:
        lines = [f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in (events or [])]
        lines.append("data: [DONE]\n\n")
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(lines).encode("utf-8"),
    )


def collect_from(handler, **kwargs):
    async def collect():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatProvider(
            base_url="https://api.deepseek.com/v1",
            api_key="secret",
            model=kwargs.pop("model", "deepseek-chat"),
            client=client,
            **kwargs,
        )
        chunks = [
            chunk
            async for chunk in provider.stream(
                system="系统", messages=[Msg.user("问题")], tools=[], usage=ModelUsage()
            )
        ]
        await client.aclose()
        return chunks

    return asyncio.run(collect())


def collect_custom(handler, *, messages=None, tools=None, **kwargs):
    async def collect():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatProvider(
            base_url="https://api.deepseek.com/v1", api_key="secret",
            model="deepseek-chat", client=client, **kwargs,
        )
        chunks = [chunk async for chunk in provider.stream(
            system="系统", messages=messages or [Msg.user("问题")],
            tools=tools or [], usage=ModelUsage())]
        await client.aclose()
        return chunks
    return asyncio.run(collect())


def test_openai_sse_text_and_usage_are_normalized():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        body = request.content.decode()
        assert '"model":"deepseek-chat"' in body.replace(" ", "")
        events = [
            {"choices": [{"delta": {"content": "你好"}}]},
            {"choices": [{"delta": {"content": "世界"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ]
        payload = "".join(
            f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events
        ) + "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload.encode("utf-8"),
        )

    async def collect():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatProvider(
            base_url="https://api.deepseek.com/v1",
            api_key="secret",
            model="deepseek-chat",
            client=client,
        )
        chunks = [
            chunk
            async for chunk in provider.stream(
                system="系统", messages=[Msg.user("问题")], tools=[], usage=ModelUsage()
            )
        ]
        await client.aclose()
        return chunks

    chunks = asyncio.run(collect())

    assert [chunk.event for chunk in chunks] == [
        StreamEvent.TEXT_DELTA,
        StreamEvent.TEXT_DELTA,
        StreamEvent.MESSAGE_END,
    ]
    assert chunks[-1].data.input_tokens == 3
    assert chunks[-1].data.output_tokens == 2


@pytest.mark.parametrize(
    ("usage", "field"),
    [
        ({"prompt_tokens": "3", "completion_tokens": 2}, "prompt_tokens"),
        ({"prompt_tokens": 3, "completion_tokens": 1.5}, "completion_tokens"),
        ({"prompt_tokens": True, "completion_tokens": 2}, "prompt_tokens"),
        ({"prompt_tokens": 3, "completion_tokens": False}, "completion_tokens"),
        ({"prompt_tokens": None, "completion_tokens": 2}, "prompt_tokens"),
    ],
)
def test_openai_malformed_usage_token_counts_are_network_error(usage, field):
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([{"choices": [{"delta": {}}], "usage": usage}])

    with pytest.raises(NetworkError, match="invalid usage"):
        collect_from(handler)


def test_openai_non_mapping_usage_is_network_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([{"choices": [{"delta": {}}], "usage": []}])

    with pytest.raises(NetworkError, match="invalid usage"):
        collect_from(handler)


def test_openai_null_usage_is_ignored_and_real_usage_still_applied():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([
            {"choices": [{"delta": {"content": "你好"}}], "usage": None},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ])

    chunks = collect_from(handler)

    assert [chunk.event for chunk in chunks] == [
        StreamEvent.TEXT_DELTA,
        StreamEvent.MESSAGE_END,
    ]
    assert chunks[-1].data.input_tokens == 3
    assert chunks[-1].data.output_tokens == 2


def test_openai_null_usage_without_a_real_chunk_defaults_to_zero():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([{"choices": [{"delta": {"content": "你好"}}], "usage": None}])

    chunks = collect_from(handler)

    assert chunks[-1].data.input_tokens == 0
    assert chunks[-1].data.output_tokens == 0


def test_openai_integer_valued_float_token_counts_are_accepted():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([
            {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 5.0, "completion_tokens": 2.0}}
        ])

    chunks = collect_from(handler)

    assert chunks[-1].data.input_tokens == 5
    assert chunks[-1].data.output_tokens == 2


def test_openai_missing_usage_token_counts_default_to_zero():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([{"choices": [{"delta": {}}], "usage": {}}])

    chunks = collect_from(handler)
    assert chunks[-1].data.input_tokens == 0
    assert chunks[-1].data.output_tokens == 0


def test_openai_provider_passes_model_parameters_and_normalizes_tool_delta():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["temperature"] == 0.25
        assert payload["max_tokens"] == 512
        return sse_response([
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_", "arguments": '{"symbol":"'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "quote", "arguments": '600519"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])

    chunks = collect_from(handler, temperature=0.25, max_tokens=512)
    assert chunks[0].data == ToolUseDelta(
        index=0, call_id="call_1", name_delta="get_", arguments_delta='{"symbol":"'
    )
    assert chunks[1].data == ToolUseDelta(index=0, name_delta="quote", arguments_delta='600519"}')
    assert chunks[-1].data.tool_uses == [ToolUse("call_1", "get_quote", {"symbol": "600519"})]


def test_openai_done_without_usage_emits_one_message_end():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(lines=["data: [DONE]\n", "\n"])

    chunks = collect_from(handler)
    assert [chunk.event for chunk in chunks] == [StreamEvent.MESSAGE_END]
    assert chunks[0].data.input_tokens == 0


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, AuthError), (403, AuthError), (429, RateLimitError), (500, ServerError)],
)
def test_openai_http_status_is_classified(status, error_type):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"error")

    with pytest.raises(error_type):
        collect_from(handler)


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        ("rate_limit_exceeded", RateLimitError),
        ("authentication_error", AuthError),
        ("permission_denied", AuthError),
        ("server_error", ServerError),
        ("invalid_request", NetworkError),
    ],
)
def test_openai_sse_error_payload_is_classified(code, error_type):
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([{"error": {"code": code, "message": "failure"}}])

    with pytest.raises(error_type):
        collect_from(handler)


def test_openai_invalid_json_is_network_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(lines=["data: {not-json}\n\n", "data: [DONE]\n\n"])

    with pytest.raises(NetworkError, match="invalid SSE JSON"):
        collect_from(handler)


def test_openai_payload_without_choices_is_network_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([{"id": "chatcmpl-x"}])

    with pytest.raises(NetworkError, match="without choices"):
        collect_from(handler)


def test_openai_other_http_4xx_is_network_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b"bad request")

    with pytest.raises(NetworkError):
        collect_from(handler)


def test_openai_httpx_error_is_wrapped_as_network_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    with pytest.raises(NetworkError, match="connection failed"):
        collect_from(handler)


def test_openai_retry_metadata_marks_status_and_connection_errors():
    async def rate_limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "2.5"})

    with pytest.raises(RateLimitError) as rate_limit:
        collect_from(rate_limited)
    assert rate_limit.value.retryable is True
    assert rate_limit.value.retry_after_s == 2.5

    async def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(ServerError) as server_error:
        collect_from(unavailable)
    assert server_error.value.retryable is True

    async def connection_failed(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    with pytest.raises(NetworkError) as network_error:
        collect_from(connection_failed)
    assert network_error.value.retryable is True


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        (60, pytest.approx(60, abs=2)),
        (-60, 0.0),
        ("not-a-delay", None),
        ("NaN", None),
        ("Infinity", None),
    ],
)
def test_openai_retry_after_supports_http_dates_and_ignores_invalid_values(retry_after, expected):
    # Date headers must be formatted at run time: a collection-time stamp
    # drifts past the assertion tolerance once the suite runs long enough.
    if isinstance(retry_after, str):
        header = retry_after
    else:
        header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=retry_after))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": header})

    with pytest.raises(RateLimitError) as error:
        collect_from(handler)

    assert error.value.retry_after_s == expected


def test_openai_protocol_and_idle_failures_are_not_retryable():
    async def invalid_payload(request: httpx.Request) -> httpx.Response:
        return sse_response(lines=["data: {bad}\n\n"])

    with pytest.raises(NetworkError) as malformed:
        collect_from(invalid_payload)
    assert malformed.value.retryable is False

    class SlowLines(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"
            await asyncio.sleep(0.05)
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            return None

    async def idle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowLines())

    with pytest.raises(NetworkError) as idle_error:
        collect_from(idle, idle_timeout_s=0.001)
    assert idle_error.value.retryable is False


def test_openai_auth_failure_is_not_retryable():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    with pytest.raises(AuthError) as error:
        collect_from(handler)
    assert error.value.retryable is False


def test_openai_eof_without_done_still_emits_message_end():
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(lines=["data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"])

    chunks = collect_from(handler)
    assert [chunk.event for chunk in chunks] == [StreamEvent.TEXT_DELTA, StreamEvent.MESSAGE_END]


def test_openai_first_byte_timeout_is_network_error():
    class SlowLines(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.05)
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            return None

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowLines())

    with pytest.raises(NetworkError, match="first response byte") as error:
        collect_from(handler, first_byte_timeout_s=0.001)
    assert error.value.retryable is True


def test_openai_idle_timeout_is_network_error():
    class SlowLines(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"
            await asyncio.sleep(0.05)
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            return None

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowLines())

    with pytest.raises(NetworkError, match="idle timeout"):
        collect_from(handler, idle_timeout_s=0.001)


def test_openai_request_schema_converts_messages_tools_and_parameters():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"] == [
            {"role": "system", "content": "系统"},
            {"role": "user", "content": "请查报价"},
            {
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_quote", "arguments": '{"symbol": "600519"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "600519: 100"},
        ]
        assert payload["tools"] == [{"type": "function", "function": {"name": "get_quote", "parameters": {"type": "object"}}}]
        return sse_response([{"choices": [{"delta": {}, "finish_reason": "stop"}]}])

    messages = [
        Msg.user("请查报价"),
        Msg(role="assistant", content="", tool_uses=[ToolUse("call_1", "get_quote", {"symbol": "600519"})]),
        Msg(role="tool_result", content=None, tool_results=[("call_1", "600519: 100")]),
    ]
    chunks = collect_custom(handler, messages=messages, tools=[{"type": "function", "function": {"name": "get_quote", "parameters": {"type": "object"}}}])
    assert chunks[-1].event == StreamEvent.MESSAGE_END


@pytest.mark.parametrize("event", [
    {"choices": {}},
    {"choices": ["bad"]},
    {"choices": [{"delta": []}]},
    {"choices": [{"delta": {"tool_calls": {}}}]},
    {"choices": [{"delta": {"tool_calls": ["bad"]}}]},
    {"choices": [{"delta": {"tool_calls": [{"function": []}]}}]},
])
def test_openai_malformed_sse_structure_is_network_error(event):
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([event])

    with pytest.raises(NetworkError):
        collect_from(handler)


@pytest.mark.parametrize("event", [
    {"choices": [{"delta": {"content": 123}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": "0"}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"id": 1}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"function": {"name": 1}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": []}}]}}]},
])
def test_openai_malformed_scalar_fields_are_network_error(event):
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([event])

    with pytest.raises(NetworkError):
        collect_from(handler)
