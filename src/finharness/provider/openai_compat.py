"""OpenAI 兼容的流式 provider。"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from finharness.provider.base import Provider
from finharness.provider.errors import (
    AuthError,
    NetworkError,
    RateLimitError,
    ServerError,
    TokenLimitError,
    is_token_limit_error,
    parse_retry_after,
)
from finharness.provider.event_stream import ToolUseAccumulator, iter_sse_data
from finharness.types import ModelUsage, Msg, StreamChunk, StreamEvent, ToolUseDelta


def _raise_provider_error(error: Any) -> None:
    """根据 provider 返回的错误内容映射为对应的 provider 异常。"""
    text = str(error).lower() if not isinstance(error, dict) else " ".join(str(error.get(k, "")) for k in ("code", "type", "message", "error")).lower()
    if is_token_limit_error(text):
        raise TokenLimitError("Provider context window exceeded")
    if any(x in text for x in ("rate_limit", "rate limit", "ratelimit")):
        raise RateLimitError("Provider rate limit exceeded")
    if any(x in text for x in ("authentication", "permission", "api_key", "api key", "unauthorized")):
        raise AuthError("Provider authentication failed")
    if "server" in text:
        raise ServerError("Provider server error")
    raise NetworkError("Provider returned an error")


async def _raise_http_error(response: Any) -> None:
    """把 4xx 响应细分为上下文超限或一般请求失败。

    超限时读取响应体并与 ``_raise_provider_error`` 用同一套特征匹配，使
    "Token 超限"能作为独立错误类型被观测到，而不是混进通用网络错误。
    """
    body = ""
    try:
        body = (await response.aread()).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 读不到正文时按一般失败处理
        body = ""
    if is_token_limit_error(body):
        raise TokenLimitError("Provider context window exceeded")
    raise NetworkError(f"Provider request failed ({response.status_code})")


def _update_usage(final: ModelUsage, raw_usage: Any) -> None:
    """用 provider 返回的原始 usage 更新最终 ModelUsage（含前缀缓存计量）。"""
    if raw_usage is None:
        # 许多 OpenAI 兼容网关在每个增量上都发送 "usage": null，
        # 仅在最后一个分块才填充；usage 缺失并非错误。
        return
    if not isinstance(raw_usage, dict):
        raise NetworkError("Provider returned invalid usage")
    values: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens"):
        value = raw_usage.get(field, 0)
        if isinstance(value, bool):
            raise NetworkError("Provider returned invalid usage token count")
        if isinstance(value, int):
            values[field] = value
        elif isinstance(value, float) and value.is_integer():
            values[field] = int(value)
        else:
            raise NetworkError("Provider returned invalid usage token count")
    final.input_tokens = values["prompt_tokens"]
    final.output_tokens = values["completion_tokens"]
    # 前缀缓存拆分，由支持该特性的 provider 报告（DeepSeek 会发送
    # prompt_cache_hit_tokens / prompt_cache_miss_tokens）。缺失表示该 provider
    # 未启用缓存，此时拆分保持为零，并非错误。
    for source, target in (
        ("prompt_cache_hit_tokens", "cache_hit_tokens"),
        ("prompt_cache_miss_tokens", "cache_miss_tokens"),
    ):
        value = raw_usage.get(source)
        if value is None:
            continue
        if isinstance(value, bool):
            raise NetworkError("Provider returned invalid cache usage count")
        if isinstance(value, int):
            pass
        elif isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise NetworkError("Provider returned invalid cache usage count")
        setattr(final, target, value)


class OpenAICompatProvider(Provider):
    def __init__(self, *, base_url: str, api_key: str, model: str, client: httpx.AsyncClient,
                 temperature: float = 0.1, max_tokens: int = 4096,
                 first_byte_timeout_s: float = 30.0, idle_timeout_s: float = 60.0):
        self.base_url, self.api_key, self.model, self.client = base_url.rstrip("/"), api_key, model, client
        self.temperature, self.max_tokens = temperature, max_tokens
        self.first_byte_timeout_s, self.idle_timeout_s = first_byte_timeout_s, idle_timeout_s

    async def stream(self, *, system: str, messages: list[Msg], tools: list[dict], usage: ModelUsage) -> AsyncIterator[StreamChunk]:
        """以流式方式请求 OpenAI 兼容的 chat completions 接口并产出规范化事件。

        负责组装请求消息（含工具调用与工具结果）、解析 SSE 增量，
        并累积工具调用片段直至流结束。
        """
        request_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            if message.role == "tool_result":
                request_messages.extend({"role": "tool", "tool_call_id": cid, "content": content} for cid, content in message.tool_results)
            elif message.tool_uses:
                request_messages.append({"role": "assistant", "content": message.content, "tool_calls": [{"id": t.call_id, "type": "function", "function": {"name": t.name, "arguments": json.dumps(t.args, ensure_ascii=False)}} for t in message.tool_uses]})
            else:
                request_messages.append({"role": message.role, "content": message.content})
        payload = {"model": self.model, "messages": request_messages, "tools": tools, "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": True, "stream_options": {"include_usage": True}}
        accumulator, final_usage = ToolUseAccumulator(), ModelUsage()
        try:
            async with self.client.stream("POST", f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=payload) as response:
                if response.status_code in (401, 403): raise AuthError(f"Provider authentication failed ({response.status_code})")
                if response.status_code == 429: raise RateLimitError("Provider rate limit exceeded", retry_after_s=parse_retry_after(response.headers.get("Retry-After")))
                if response.status_code >= 500: raise ServerError(f"Provider server error ({response.status_code})")
                if response.status_code >= 400:
                    await _raise_http_error(response)
                async for data in iter_sse_data(response.aiter_lines(), first_byte_timeout_s=self.first_byte_timeout_s, idle_timeout_s=self.idle_timeout_s):
                    if data.strip() == "[DONE]": break
                    try: event = json.loads(data)
                    except json.JSONDecodeError as exc: raise NetworkError("Provider returned invalid SSE JSON") from exc
                    if not isinstance(event, dict): raise NetworkError("Provider returned invalid SSE payload")
                    if event.get("error") is not None: _raise_provider_error(event["error"])
                    usage_present = "usage" in event
                    event_usage = event.get("usage")
                    if usage_present:
                        _update_usage(final_usage, event_usage)
                    choices = event.get("choices")
                    if choices is None:
                        if not usage_present: raise NetworkError("Provider returned payload without choices")
                        continue
                    if not isinstance(choices, list):
                        raise NetworkError("Provider returned invalid choices")
                    if not choices:
                        if not usage_present: raise NetworkError("Provider returned payload without choices")
                        continue
                    if not isinstance(choices[0], dict):
                        raise NetworkError("Provider returned invalid choice")
                    delta = choices[0].get("delta", {})
                    if delta is None:
                        delta = {}
                    if not isinstance(delta, dict):
                        raise NetworkError("Provider returned invalid delta")
                    content = delta.get("content")
                    if content is not None and not isinstance(content, str):
                        raise NetworkError("Provider returned invalid content")
                    if content:
                        yield StreamChunk(StreamEvent.TEXT_DELTA, content)
                    tool_calls = delta.get("tool_calls", [])
                    if tool_calls is None:
                        tool_calls = []
                    if not isinstance(tool_calls, list):
                        raise NetworkError("Provider returned invalid tool_calls")
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            raise NetworkError("Provider returned invalid tool call")
                        index = tc.get("index", 0)
                        if not isinstance(index, int) or isinstance(index, bool):
                            raise NetworkError("Provider returned invalid tool call index")
                        call_id = tc.get("id")
                        if call_id is not None and not isinstance(call_id, str):
                            raise NetworkError("Provider returned invalid tool call id")
                        fn = tc.get("function", {}) or {}
                        if not isinstance(fn, dict):
                            raise NetworkError("Provider returned invalid tool function")
                        name = fn.get("name", "")
                        arguments = fn.get("arguments", "")
                        if name is None:
                            name = ""
                        if arguments is None:
                            arguments = ""
                        if not isinstance(name, str) or not isinstance(arguments, str):
                            raise NetworkError("Provider returned invalid tool function fields")
                        td = ToolUseDelta(index=index, call_id=call_id, name_delta=name, arguments_delta=arguments)
                        accumulator.add(td)
                        yield StreamChunk(StreamEvent.TOOL_USE_DELTA, td)
        except httpx.HTTPError as exc:
            retryable = isinstance(
                exc,
                (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.WriteError),
            )
            raise NetworkError(str(exc), retryable=retryable) from exc
        final_usage.tool_uses = accumulator.build()
        yield StreamChunk(StreamEvent.MESSAGE_END, final_usage)
