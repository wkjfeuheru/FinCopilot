"""MCP 客户端：握手、会话、SSE 解析与故障分类。

全程使用 ``httpx.MockTransport``，因此这里不触碰网络。这些测试是本集成里唯一
能把同花顺 MCP 的线协议钉住的地方——公开文档没有写握手细节，所以"客户端对
标准 Streamable HTTP 的行为"只能由这里的断言来保证。
"""

from __future__ import annotations

import json

import httpx
import pytest

from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.mcp_client import (
    PROTOCOL_VERSION,
    McpHttpClient,
    McpToolError,
)

URL = "https://fuyao.aicubes.cn/mcp/a-share"


def ok(payload: dict, request_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def ack(body: dict, payload: dict | None = None, **kwargs) -> httpx.Response:
    """按请求体构造响应；通知（无 ``id``）以 202 应答。

    客户端会发送一条无 id 的 ``notifications/initialized``，而通知按 JSON-RPC 规范
    不需要响应体——因此每个 responder 都必须能应付"没有 id 的请求"。
    """
    if body.get("id") is None:
        return httpx.Response(202, **kwargs)
    return httpx.Response(200, json=ok(payload or {}, request_id=body["id"]), **kwargs)


def envelope_ok(payload: dict) -> dict:
    return {"code": 0, "message": "ok", "request_id": "r", "data": payload}


class Recorder:
    """记录每次请求，便于断言"发了什么、发了几次"。"""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []

    def methods(self) -> list[str]:
        return [body.get("method") for body in self.bodies]


def make_client(handler, *, api_key: str | None = "k") -> McpHttpClient:
    return McpHttpClient(
        url=URL,
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def recording(responder) -> tuple[McpHttpClient, Recorder]:
    """构造一个客户端，其服务端按 ``responder(request, body)`` 应答。"""
    rec = Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        body = json.loads(request.read().decode())
        rec.bodies.append(body)
        return responder(request, body)

    return make_client(handler), rec


# -- 握手与会话 ------------------------------------------------------------


def test_initialize_runs_once_and_the_session_id_is_echoed_back():
    """标准 Streamable HTTP：先 initialize 取会话 id，后续每个请求都带回它。"""
    seen_sessions: list[str | None] = []

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        if body["method"] == "initialize":
            return ack(body, {"protocolVersion": PROTOCOL_VERSION}, headers={"Mcp-Session-Id": "sess-1"})
        if body["method"] == "tools/list":
            seen_sessions.append(request.headers.get("Mcp-Session-Id"))
        return ack(body, {"tools": []})

    client, rec = recording(responder)
    client.list_tools()
    client.list_tools()

    assert rec.methods() == ["initialize", "notifications/initialized", "tools/list"]
    assert seen_sessions == ["sess-1"]


def test_initialize_announces_the_protocol_version_afterwards():
    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return ack(body, {"protocolVersion": "2025-03-26"}, headers={"Mcp-Session-Id": "s"})

    client, rec = recording(responder)
    client.list_tools()

    # 服务端公布了它支持的版本时，以服务端为准。
    assert rec.requests[-1].headers.get("MCP-Protocol-Version") == "2025-03-26"


def test_a_stateless_server_is_handshaked_only_once():
    """服务端不返回会话 id 是合法的；不能因此每次调用都重新 initialize。"""

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        if body["method"] == "tools/list":
            return ack(body, {"tools": []})
        return ack(body)

    client, rec = recording(responder)
    client.list_tools()
    client.list_tools()

    assert rec.methods().count("initialize") == 1
    assert rec.methods().count("tools/list") == 1  # 第二次命中目录缓存


def test_the_api_key_is_sent_on_every_request_including_initialize():
    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return ack(body, {"tools": []})

    client, rec = recording(responder)
    client.list_tools()

    assert all(r.headers.get("X-api-key") == "k" for r in rec.requests)


def test_a_lost_session_is_rebuilt_and_the_call_retried_once():
    """服务端重启过：重建会话即可，不需要把这次失败上报给调用方。"""
    state = {"lists": 0, "initializes": 0}

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        if body["method"] == "initialize":
            state["initializes"] += 1
            return ack(body, {}, headers={"Mcp-Session-Id": f"s{state['initializes']}"})
        if body["method"] != "tools/list":
            # 通知：不该消耗"第一次 tools/list 会失败"这个设定。
            return ack(body)
        state["lists"] += 1
        if state["lists"] == 1:
            return httpx.Response(404, text="session not found")
        return ack(body, {"tools": [{"name": "t"}]})

    client, rec = recording(responder)
    tools = client.list_tools()

    assert [t["name"] for t in tools] == ["t"]
    assert state["initializes"] == 2
    assert rec.methods().count("tools/list") == 2


# -- 结果解析 --------------------------------------------------------------


def test_structured_content_is_returned_directly():
    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return ack(
            body,
            {"structuredContent": envelope_ok({"item": [{"ticker": "600519"}]})},
        )

    client, _ = recording(responder)

    assert client.call_tool("get_a_share_prices_snapshot", {})["code"] == 0


def test_a_json_text_block_is_parsed_into_a_payload():
    """上游把统一响应信封整体序列化进 content[] 的文本块，因此要解开它。"""
    envelope = envelope_ok({"item": [{"ticker": "000858"}]})

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return ack(body, {"content": [{"type": "text", "text": json.dumps(envelope)}]})

    client, _ = recording(responder)

    assert client.call_tool("x", {})["data"]["item"][0]["ticker"] == "000858"


def test_a_non_json_text_block_is_preserved_as_text():
    """不是 JSON 就原样保留，而不是丢掉内容。"""

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return ack(body, {"content": [{"type": "text", "text": "无数据"}]})

    client, _ = recording(responder)

    assert client.call_tool("x", {}) == {"text": "无数据"}


def test_an_error_content_block_is_a_tool_error():
    """``isError=true`` 是工具级失败，与传输失败的可重试性不同。"""

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return ack(body, {"isError": True, "content": [{"type": "text", "text": "code=1002 参数错误"}]})

    client, _ = recording(responder)

    with pytest.raises(McpToolError) as exc:
        client.call_tool("x", {"thscode": "600519.SH,000858.SZ"})

    assert "1002" in str(exc.value)
    assert exc.value.retryable is False


# -- SSE -------------------------------------------------------------------


def test_an_sse_response_is_parsed():
    """Streamable HTTP 允许 SSE 应答；选哪条是服务端的自由，两条都要支持。"""
    event = json.dumps(ok({"tools": [{"name": "get_a_share_prices_snapshot"}]}, request_id=2))

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        if body["method"] == "initialize":
            return ack(body)
        return httpx.Response(
            200,
            text="event: message\ndata: " + event + "\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client, _ = recording(responder)

    assert client.list_tools()[0]["name"] == "get_a_share_prices_snapshot"


def test_sse_frames_that_are_not_ours_are_skipped():
    """事件流里可能夹着通知或其它请求的响应；本次调用等的是自己的 id。"""
    noise = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"})
    mine = json.dumps(ok({"tools": []}, request_id=2))

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        if body["method"] == "initialize":
            return ack(body)
        return httpx.Response(
            200,
            text="data: " + noise + "\n\ndata: " + mine + "\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client, _ = recording(responder)

    assert client.list_tools() == []


def test_a_corrupt_sse_frame_does_not_fail_the_whole_response():
    event = json.dumps(ok({"tools": [{"name": "t"}]}, request_id=2))

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        if body["method"] == "initialize":
            return ack(body)
        return httpx.Response(
            200,
            text="data: {not json\n\ndata: " + event + "\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client, _ = recording(responder)

    assert [t["name"] for t in client.list_tools()] == ["t"]


# -- 故障分类 --------------------------------------------------------------


def test_a_json_rpc_error_is_a_tool_error():
    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32602, "message": "Invalid params", "data": "thscode"},
            },
        )

    client, _ = recording(responder)

    with pytest.raises(McpToolError) as exc:
        client.list_tools()

    assert "-32602" in str(exc.value)
    assert "thscode" in str(exc.value)


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (500, True), (503, True), (401, False), (403, False)],
)
def test_http_status_maps_to_the_right_retryability(status, retryable):
    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return httpx.Response(status, text="boom")

    client, _ = recording(responder)

    with pytest.raises(AdapterError) as exc:
        client.list_tools()

    assert exc.value.retryable is retryable


def test_a_network_failure_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(AdapterError) as exc:
        make_client(handler).list_tools()

    assert exc.value.retryable is True


def test_a_missing_key_fails_before_any_request_is_sent():
    """把它发出去只会得到 401，而错误信息不如本地这条清楚。"""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=ok({"tools": []}))

    client = make_client(handler, api_key=None)

    with pytest.raises(AdapterError) as exc:
        client.list_tools()

    assert sent == []
    assert "未配置" in str(exc.value)
    assert exc.value.retryable is False


def test_a_non_json_response_is_reported_with_its_content_type():
    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return httpx.Response(200, text="<html>oops</html>")

    client, _ = recording(responder)

    with pytest.raises(AdapterError) as exc:
        client.list_tools()

    assert "非 JSON" in str(exc.value)


def test_a_response_without_a_matching_id_is_an_error():
    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        return httpx.Response(200, json=ok({"tools": []}, request_id=9999))

    client, _ = recording(responder)

    with pytest.raises(AdapterError) as exc:
        client.list_tools()

    assert "没有匹配的结果" in str(exc.value)


# -- 目录缓存 --------------------------------------------------------------


def test_the_tool_catalog_is_cached_per_client():
    calls = {"n": 0}

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        if body["method"] == "tools/list":
            calls["n"] += 1
        return ack(body, {"tools": [{"name": "t"}]})

    client, _ = recording(responder)
    client.list_tools()
    client.list_tools()

    assert calls["n"] == 1


def test_invalidate_forces_a_fresh_handshake_and_catalog():
    calls: list[str] = []

    def responder(request: httpx.Request, body: dict) -> httpx.Response:
        calls.append(body["method"])
        return ack(body, {"tools": [{"name": "t"}]}, headers={"Mcp-Session-Id": "s"})

    client, _ = recording(responder)
    client.list_tools()
    client.invalidate()
    client.list_tools()

    assert calls.count("initialize") == 2
    assert calls.count("tools/list") == 2
