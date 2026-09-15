"""Anthropic Messages 兼容流式 Provider。"""
from __future__ import annotations
import json
from collections.abc import AsyncIterator
from typing import Any
import httpx
from finharness.provider.base import Provider
from finharness.provider.errors import AuthError, NetworkError, RateLimitError, ServerError, TokenLimitError, is_token_limit_error, parse_retry_after
from finharness.provider.event_stream import ToolUseAccumulator, iter_sse_data
from finharness.types import ModelUsage, Msg, StreamChunk, StreamEvent, ToolUseDelta

def _raise_error(error: Any) -> None:
    """根据 provider 返回的错误内容映射为对应的 provider 异常。"""
    text = str(error).lower() if not isinstance(error, dict) else " ".join(str(error.get(k, "")) for k in ("type", "code", "message", "error")).lower()
    if is_token_limit_error(text): raise TokenLimitError("Provider context window exceeded")
    if any(t in text for t in ("rate_limit", "rate limit", "ratelimit")): raise RateLimitError("Provider rate limit exceeded")
    if any(t in text for t in ("authentication", "permission", "api_key", "api key", "unauthorized")): raise AuthError("Provider authentication failed")
    if "server" in text: raise ServerError("Provider server error")
    raise NetworkError("Provider returned an error")

async def _raise_http_error(response: Any) -> None:
    """把 4xx 响应细分为上下文超限或一般请求失败（与 OpenAI 兼容层一致）。"""
    try:
        body = (await response.aread()).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 读不到正文时按一般失败处理
        body = ""
    if is_token_limit_error(body): raise TokenLimitError("Provider context window exceeded")
    raise NetworkError(f"Provider request failed ({response.status_code})")

def _convert_tools(tools: list[dict]) -> list[dict]:
    """将函数工具 schema 转换为 Anthropic ``tools`` 格式。"""
    out = []
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str): raise NetworkError("Invalid tool schema")
        params = fn.get("parameters", {"type": "object"})
        if not isinstance(params, dict): raise NetworkError("Invalid tool schema")
        out.append({"name": fn["name"], "description": fn.get("description", "") or "", "input_schema": params})
    return out

def _convert_messages(messages: list[Msg]) -> list[dict]:
    """将规范化消息转换为 Anthropic Messages 接口所需的消息块列表。"""
    out = []
    for m in messages:
        if m.role == "tool_result":
            blocks = [{"type": "tool_result", "tool_use_id": cid, "content": c} for cid, c in m.tool_results]
            if blocks: out.append({"role": "user", "content": blocks})
        elif m.tool_uses:
            blocks = ([{"type": "text", "text": m.content}] if m.content else [])
            blocks += [{"type": "tool_use", "id": t.call_id, "name": t.name, "input": t.args} for t in m.tool_uses]
            out.append({"role": "assistant", "content": blocks})
        else: out.append({"role": m.role, "content": m.content or ""})
    return out


def _update_usage(final: ModelUsage, raw_usage: Any) -> None:
    """用 provider 返回的原始 usage 更新最终 ModelUsage（含 prompt 缓存计量）。"""
    if not isinstance(raw_usage, dict):
        raise NetworkError("Provider returned invalid usage")
    for field in ("input_tokens", "output_tokens"):
        value = raw_usage.get(field)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise NetworkError("Provider returned invalid usage token count")
        setattr(final, field, value)
    # prompt 缓存计量。Anthropic 将输入拆分为从缓存读取与写入缓存两部分；
    # 只有在启用缓存时才会报告这两项，因此字段缺失是正常情况而非错误。
    for source, target in (
        ("cache_read_input_tokens", "cache_hit_tokens"),
        ("cache_creation_input_tokens", "cache_miss_tokens"),
    ):
        value = raw_usage.get(source)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise NetworkError("Provider returned invalid cache usage count")
        setattr(final, target, value)

class AnthropicCompatProvider(Provider):
    def __init__(self, *, base_url: str, api_key: str, model: str, api_version: str = "2023-06-01", temperature: float = 0.1, max_tokens: int = 4096, client: httpx.AsyncClient, first_byte_timeout_s: float = 30.0, idle_timeout_s: float = 60.0):
        self.base_url, self.api_key, self.model, self.api_version = base_url.rstrip("/"), api_key, model, api_version
        self.temperature, self.max_tokens, self.client = temperature, max_tokens, client
        self.first_byte_timeout_s, self.idle_timeout_s = first_byte_timeout_s, idle_timeout_s

    async def stream(self, *, system: str, messages: list[Msg], tools: list[dict], usage: ModelUsage) -> AsyncIterator[StreamChunk]:
        """以流式方式请求 Anthropic Messages 接口并产出规范化 StreamChunk 事件。"""
        payload = {"model": self.model, "system": system, "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": True, "messages": _convert_messages(messages), "tools": _convert_tools(tools)}
        acc, final = ToolUseAccumulator(), ModelUsage(); seen: set[int] = set()
        try:
            async with self.client.stream("POST", f"{self.base_url}/messages", headers={"x-api-key": self.api_key, "anthropic-version": self.api_version, "content-type": "application/json"}, json=payload) as response:
                if response.status_code in (401, 403): raise AuthError(f"Provider authentication failed ({response.status_code})")
                if response.status_code == 429: raise RateLimitError("Provider rate limit exceeded", retry_after_s=parse_retry_after(response.headers.get("Retry-After")))
                if response.status_code >= 500: raise ServerError(f"Provider server error ({response.status_code})")
                if response.status_code >= 400: await _raise_http_error(response)
                async for data in iter_sse_data(response.aiter_lines(), first_byte_timeout_s=self.first_byte_timeout_s, idle_timeout_s=self.idle_timeout_s):
                    try: event = json.loads(data)
                    except json.JSONDecodeError as exc: raise NetworkError("Provider returned invalid SSE JSON") from exc
                    if not isinstance(event, dict): raise NetworkError("Provider returned invalid SSE payload")
                    if event.get("type") == "error" or event.get("error") is not None: _raise_error(event.get("error", event))
                    typ = event.get("type")
                    if not isinstance(typ, str):
                        raise NetworkError("Provider returned invalid SSE event type")
                    if typ == "message_start":
                        message = event.get("message")
                        if not isinstance(message, dict):
                            raise NetworkError("Provider returned invalid message_start")
                        _update_usage(final, message.get("usage", {}))
                    elif typ == "message_delta":
                        _update_usage(final, event.get("usage", {}))
                    elif typ == "content_block_start":
                        i, b = event.get("index"), event.get("content_block")
                        if not isinstance(i, int) or isinstance(i, bool) or not isinstance(b, dict): raise NetworkError("Provider returned invalid content block")
                        if b.get("type") == "tool_use":
                            if not isinstance(b.get("id"), str) or not isinstance(b.get("name"), str) or not isinstance(b.get("input", {}), dict): raise NetworkError("Provider returned invalid tool use")
                            d = ToolUseDelta(i, b["id"], b["name"], json.dumps(b.get("input", {}), ensure_ascii=False) if b.get("input") else ""); acc.add(d); yield StreamChunk(StreamEvent.TOOL_USE_DELTA, d)
                        elif b.get("type") == "text":
                            if "text" in b and not isinstance(b.get("text"), str):
                                raise NetworkError("Provider returned invalid text block")
                        else:
                            raise NetworkError("Provider returned invalid content block")
                    elif typ == "content_block_delta":
                        i, d = event.get("index"), event.get("delta")
                        if not isinstance(i, int) or isinstance(i, bool) or not isinstance(d, dict): raise NetworkError("Provider returned invalid content block delta")
                        delta_type = d.get("type")
                        if not isinstance(delta_type, str):
                            raise NetworkError("Provider returned invalid content block delta")
                        if delta_type == "text_delta":
                            if not isinstance(d.get("text", ""), str): raise NetworkError("Provider returned invalid text delta")
                            if d.get("text"): yield StreamChunk(StreamEvent.TEXT_DELTA, d["text"])
                        elif delta_type == "input_json_delta":
                            frag = d.get("partial_json", "")
                            if not isinstance(frag, str): raise NetworkError("Provider returned invalid input JSON delta")
                            if i not in seen: acc.replace_arguments(i); seen.add(i)
                            td = ToolUseDelta(i, arguments_delta=frag); acc.add(td); yield StreamChunk(StreamEvent.TOOL_USE_DELTA, td)
                        elif delta_type not in {"text_delta", "input_json_delta"}:
                            raise NetworkError("Provider returned invalid content block delta")
                    elif typ == "message_stop": break
                    elif typ in {"ping", "content_block_stop"}:
                        continue
                    elif typ not in {"message_start", "message_delta", "content_block_start", "content_block_delta"}:
                        raise NetworkError("Provider returned invalid SSE event")
        except httpx.HTTPError as exc:
            retryable = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.WriteError))
            raise NetworkError(str(exc), retryable=retryable) from exc
        final.tool_uses = acc.build(); yield StreamChunk(StreamEvent.MESSAGE_END, final)
