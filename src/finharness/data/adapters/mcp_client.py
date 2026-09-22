"""最小 MCP 客户端：JSON-RPC 2.0 over Streamable HTTP（docs 03.5）。

**为什么是手写的。** 本项目的每条出站数据路径都是一个同步 ``httpx`` 客户端
（``TavilyAdapter``、``EastmoneyReportAdapter``、``pdf_fetch``），由
``DataAccess`` 经 ``asyncio.to_thread`` 调用。官方 ``mcp`` SDK 只提供异步 API，
接入它就得引入"后台线程 + 常驻事件循环"这一本项目此前没有的结构，只为把异步
包成同步。协议本身只用到四个方法（initialize / notifications/initialized /
tools/list / tools/call），因此这里实现这个子集，代价可控、无新增依赖。

**已知的不确定性。** 同花顺的公开文档只说明客户端配置为 ``"type": "http"`` 加
``X-api-key`` 请求头，**没有写握手细节**。因此本客户端的目标是"两种服务端都
兼容"：它先尝试标准 Streamable HTTP 握手（``initialize`` → 取
``Mcp-Session-Id`` → ``notifications/initialized``），但只在**首次**调用时做；
服务端若不返回会话 id（无状态直调），后续调用就照常进行而不再重复握手。
协议细节全部隔离在本模块内，因此若线上行为不符，替换成本局限于此文件。

**响应可能是 JSON 也可能是 SSE。** Streamable HTTP 允许服务端用
``text/event-stream`` 应答，同一个调用可能被拆成若干事件。两条路径都要解析，
因为服务端选哪条并不由客户端决定。
"""

from __future__ import annotations

import json
import threading
from typing import Any

import httpx

from finharness.data.adapters.base import AdapterError

# MCP 协议版本。日期形式的版本号是 MCP 的约定；服务端若在 initialize 中返回它
# 支持的版本，以服务端为准。
PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "finharness"
CLIENT_VERSION = "0.1.0"

_API_KEY_HEADER = "X-api-key"
_SESSION_HEADER = "Mcp-Session-Id"
_PROTOCOL_HEADER = "MCP-Protocol-Version"
_ACCEPT = "application/json, text/event-stream"

# 会话失效的上游信号：服务端不认识我们持有的 session id。重建一次并重试即可——
# 这是"服务端重启过"的正常结果，不是需要上报给用户的错误。
_SESSION_LOST_STATUSES = frozenset({400, 404})


def _error_message(payload: Any) -> str:
    """从 JSON-RPC 错误对象里取可读信息。"""
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message") or ""
        detail = f"{code}: {message}" if code is not None else str(message)
        data = error.get("data")
        if data not in (None, "", {}):
            detail = f"{detail}（{data}）"
        return detail
    return str(payload)[:300]


def _content_text(result: dict[str, Any]) -> str:
    """把 MCP ``content[]`` 块里的文本拼接起来。"""
    parts: list[str] = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _parse_sse(text: str) -> list[dict[str, Any]]:
    """解析 SSE 事件流，收集其中每条 ``data:`` 载荷的 JSON。

    SSE 帧以空行分隔，字段形如 ``event: message`` / ``data: {...}``。多行 ``data``
    按规范以换行拼接；注释行（``:`` 开头）与 ``event``/``id`` 字段在此无关紧要，
    因为 MCP 只用 ``data`` 承载消息。
    """
    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []

    def flush() -> None:
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        data_lines.clear()
        try:
            parsed = json.loads(raw)
        except ValueError:
            # 单个坏帧不该让整个响应失败：其余帧仍可能带着我们等的结果。
            return
        messages.extend(parsed if isinstance(parsed, list) else [parsed])

    for line in text.splitlines():
        if not line.strip():
            flush()
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field.strip() == "data":
            data_lines.append(value.lstrip(" "))
    flush()
    return messages


class McpToolError(AdapterError):
    """工具本身返回的错误（``isError=true`` 或 JSON-RPC error）。

    与传输层失败分开，是因为两者的可重试性不同：一次参数错误重试多少次都一样，
    而连接重置值得再试一次。
    """


class _SessionLost(AdapterError):
    """内部信号：服务端不认识当前会话 id。"""


class McpHttpClient:
    """一个 MCP 服务端（一个 URL）的同步客户端。

    会话状态是**每个实例**的：六个同花顺服务各有各的连接，共用实例会让后者的
    initialize 覆盖前者的会话 id。
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        timeout_s: float = 30.0,
        proxy: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.proxy = proxy
        # 可注入，使测试用 ``httpx.MockTransport`` 覆盖全部解析分支而不触碰网络。
        self._client = client
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        # 与 ``_session_id`` 分开：无状态服务端从不返回会话 id，若用它判断
        # "是否已握手"，每次调用都会重新 initialize。
        self._initialized = False
        self._request_id = 0
        # ``DataAccess`` 通过 ``asyncio.to_thread`` 调用适配器，因此抓取发生在工作
        # 线程而非事件循环线程；会话的读写与"失效即重建"必须互斥，否则一次并发
        # 调用可能同时初始化出两个会话，其中一个的 id 立刻被另一个覆盖。
        self._lock = threading.Lock()
        self._tools_cache: list[dict[str, Any]] | None = None

    # -- 传输层 ---------------------------------------------------------------
    def _headers(self, *, with_session: bool) -> dict[str, str]:
        if not self.api_key:
            # 在发出请求前失败：把它发出去只会得到一个 401，而错误信息不如这里清楚。
            raise AdapterError(
                "未配置同花顺密钥（HITHINK_FINANCE_API_KEY）：同花顺数据源当前不可用"
            )
        headers = {
            "Content-Type": "application/json",
            "Accept": _ACCEPT,
            _API_KEY_HEADER: self.api_key,
        }
        if with_session and self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        if self._protocol_version:
            headers[_PROTOCOL_HEADER] = self._protocol_version
        return headers

    def _post(self, body: dict[str, Any], *, with_session: bool) -> httpx.Response:
        headers = self._headers(with_session=with_session)
        try:
            if self._client is not None:
                return self._client.post(
                    self.url, json=body, headers=headers, timeout=self.timeout_s
                )
            with httpx.Client(timeout=self.timeout_s, proxy=self.proxy) as client:
                return client.post(self.url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            # 连接重置、TLS 抖动、超时：都值得再试一次。
            raise AdapterError(f"同花顺 MCP 请求失败：{exc}", retryable=True) from exc

    @staticmethod
    def _decode_messages(response: httpx.Response) -> list[dict[str, Any]]:
        """把一个响应解析为若干 JSON-RPC 消息。

        服务端可以回一个 JSON 对象，也可以回一段 SSE 事件流（每个 ``data:`` 行是
        一条消息）。两者都要处理——选哪条是服务端的自由。
        """
        content_type = (response.headers.get("content-type") or "").lower()
        text = response.text
        if "text/event-stream" in content_type:
            return _parse_sse(text)
        if not text.strip():
            return []
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise AdapterError(
                f"同花顺 MCP 返回了非 JSON 响应（content-type={content_type or '未声明'}）"
            ) from exc
        return payload if isinstance(payload, list) else [payload]

    def _parse_response(self, response: httpx.Response, *, request_id: int) -> dict[str, Any]:
        """按状态码与 JSON-RPC 信封校验一次调用的结果。"""
        if response.status_code >= 400 and self._session_id:
            # 持有会话时遇到的 400/404 意味着会话已失效；无会话时它们只是普通错误。
            if response.status_code in _SESSION_LOST_STATUSES:
                raise _SessionLost(f"会话已失效（HTTP {response.status_code}）")
        if response.status_code == 429:
            raise AdapterError("同花顺限流（HTTP 429）", retryable=True)
        if response.status_code >= 500:
            raise AdapterError(f"同花顺服务端错误（HTTP {response.status_code}）", retryable=True)
        if response.status_code >= 400:
            # 401/403 是凭据问题（同花顺文档：可能表现为 401/403 或 code=2003），
            # 重试不会让它变好。
            raise AdapterError(
                f"同花顺返回 HTTP {response.status_code}：{response.text[:200]}"
            )

        for message in self._decode_messages(response):
            if not isinstance(message, dict):
                continue
            if message.get("id") is not None and message.get("id") != request_id:
                # 通知或其它请求的响应；本次调用等的是自己的 id。
                continue
            if "error" in message:
                raise McpToolError(f"同花顺调用失败：{_error_message(message)}")
            if "result" in message:
                return message["result"]
        raise AdapterError("同花顺 MCP 响应中没有匹配的结果")

    # -- 协议层 ---------------------------------------------------------------
    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _initialize(self) -> None:
        """握手并记录会话 id。

        服务端若不返回 ``Mcp-Session-Id``（无状态模式），后续请求就不带它——这是
        合法的，因此这里不把缺席当作错误。
        """
        request_id = self._next_id()
        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        }
        response = self._post(body, with_session=False)
        result = self._parse_response(response, request_id=request_id)
        session_id = response.headers.get(_SESSION_HEADER)
        if session_id:
            self._session_id = session_id
        protocol = (result or {}).get("protocolVersion") if isinstance(result, dict) else None
        self._protocol_version = protocol or PROTOCOL_VERSION
        self._initialized = True
        self._notify_initialized()

    def _notify_initialized(self) -> None:
        """发送 ``notifications/initialized``；服务端若不认，容忍失败。

        这是 JSON-RPC 通知（无 id、无响应）。部分实现在此返回 202/204，也有实现
        直接忽略；两种都不应让初始化失败。
        """
        body = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        try:
            self._post(body, with_session=True)
        except AdapterError:
            # 通知失败不影响后续调用：真正需要会话的是 tools/call。
            pass

    def _call_locked(self, method: str, params: dict[str, Any] | None) -> Any:
        """在持有锁的前提下完成一次调用；必要时先握手，会话失效则重建一次重试。"""
        if not self._initialized:
            # 凭据缺席在此抛出，使"没配 key"在第一次调用就被发现。
            self._headers(with_session=False)
            self._initialize()
        request_id = self._next_id()
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            body["params"] = params
        response = self._post(body, with_session=True)
        try:
            return self._parse_response(response, request_id=request_id)
        except _SessionLost:
            # 服务端重启过：重建会话后重试一次。仅持有会话时才可能走到这里，
            # 因此不会与"服务端根本不要求握手"的情形互相递归。
            self._session_id = None
            self._initialized = False
            self._tools_cache = None
            self._initialize()
            retry_id = self._next_id()
            body["id"] = retry_id
            retried = self._post(body, with_session=True)
            return self._parse_response(retried, request_id=retry_id)

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            return self._call_locked(method, params)

    # -- 语义层 ---------------------------------------------------------------
    def list_tools(self) -> list[dict[str, Any]]:
        """该服务的工具清单（含 ``inputSchema``），按实例缓存。"""
        with self._lock:
            if self._tools_cache is not None:
                return self._tools_cache
        result = self._call("tools/list", {})
        tools = (result or {}).get("tools") if isinstance(result, dict) else None
        # 目录可能被一次并发调用同时写两次；两次内容相同，后写者胜出即可。
        with self._lock:
            self._tools_cache = list(tools) if isinstance(tools, list) else []
            return self._tools_cache

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用一个工具并返回其 ``structuredContent``（或解析后的文本载荷）。"""
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            return result
        if result.get("isError"):
            raise McpToolError(f"同花顺工具 {name} 返回错误：{_content_text(result)}")
        if "structuredContent" in result:
            return result["structuredContent"]
        # 没有 structuredContent 时，载荷在 content[] 的文本块里；它通常是一段
        # JSON 字符串——上游把统一响应信封整体序列化进文本。
        text = _content_text(result)
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError:
            # 不是 JSON 就原样返回文本，交由调用方渲染，而不是丢掉内容。
            return {"text": text}

    def invalidate(self) -> None:
        """丢弃会话与工具目录缓存，使下次调用重新握手。"""
        with self._lock:
            self._session_id = None
            self._initialized = False
            self._tools_cache = None
