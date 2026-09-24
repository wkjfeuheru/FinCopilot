"""两种 provider 协议共享的错误归一化与 HTTP 状态处理。

``openai_compat`` 与 ``anthropic_compat`` 曾各自实现一遍同一套逻辑：把错误正文
映射为 provider 异常、把 HTTP 状态码阶梯细分为鉴权/限流/服务端/请求失败、判断
httpx 异常是否可重试。协议差异只在"消息与工具的形状"上，错误处理是完全一致的，
因此集中到这里，避免错误串与判定顺序在两处漂移。
"""

from __future__ import annotations

from typing import Any

import httpx

from finharness.provider.errors import (
    AuthError,
    NetworkError,
    RateLimitError,
    ServerError,
    TokenLimitError,
    is_token_limit_error,
    parse_retry_after,
)

# 错误字典里按这些键拼接出可匹配的文本；顺序不影响子串判定。
_ERROR_KEYS = ("code", "type", "message", "error")


def raise_provider_error(error: Any) -> None:
    """根据 provider 返回的错误内容映射为对应的 provider 异常。"""
    if isinstance(error, dict):
        text = " ".join(str(error.get(key, "")) for key in _ERROR_KEYS).lower()
    else:
        text = str(error).lower()
    if is_token_limit_error(text):
        raise TokenLimitError("Provider context window exceeded")
    if any(marker in text for marker in ("rate_limit", "rate limit", "ratelimit")):
        raise RateLimitError("Provider rate limit exceeded")
    if any(
        marker in text
        for marker in ("authentication", "permission", "api_key", "api key", "unauthorized")
    ):
        raise AuthError("Provider authentication failed")
    if "server" in text:
        raise ServerError("Provider server error")
    raise NetworkError("Provider returned an error")


async def raise_http_error(response: Any) -> None:
    """把 4xx 响应细分为上下文超限或一般请求失败。

    超限时读取响应体并与 ``raise_provider_error`` 用同一套特征匹配，使
    "Token 超限"能作为独立错误类型被观测到，而不是混进通用网络错误。
    """
    try:
        body = (await response.aread()).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 读不到正文时按一般失败处理
        body = ""
    if is_token_limit_error(body):
        raise TokenLimitError("Provider context window exceeded")
    raise NetworkError(f"Provider request failed ({response.status_code})")


async def raise_for_status(response: Any) -> None:
    """把非 2xx 状态码映射为对应的 provider 异常（流式响应头阶段）。"""
    if response.status_code in (401, 403):
        raise AuthError(f"Provider authentication failed ({response.status_code})")
    if response.status_code == 429:
        raise RateLimitError(
            "Provider rate limit exceeded",
            retry_after_s=parse_retry_after(response.headers.get("Retry-After")),
        )
    if response.status_code >= 500:
        raise ServerError(f"Provider server error ({response.status_code})")
    if response.status_code >= 400:
        await raise_http_error(response)


def as_network_error(exc: httpx.HTTPError) -> NetworkError:
    """把 httpx 异常包装为 provider 的 ``NetworkError``，并标记是否可重试。"""
    retryable = isinstance(
        exc,
        (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.WriteError),
    )
    return NetworkError(str(exc), retryable=retryable)
