"""Provider 错误分类与重试元数据。"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite


class ProviderError(RuntimeError):
    """provider 失败，并携带供 engine 使用的重试信息。"""

    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = self.default_retryable if retryable is None else retryable
        self.retry_after_s = retry_after_s


class AuthError(ProviderError):
    default_retryable = False


class RateLimitError(ProviderError):
    default_retryable = True


class ServerError(ProviderError):
    default_retryable = True


class NetworkError(ProviderError):
    default_retryable = False


class TokenLimitError(ProviderError):
    """输入超出模型上下文窗口；重试同样的请求只会再次失败。

    单列出来是为了让可观测性能够把它与一般的 provider 失败区分开
    （docs 03.14.2：``agent_request_errors_total`` 按错误类型分别统计）。
    """

    default_retryable = False


# 各 provider 对"上下文超限"的措辞不同，因此按特征片段匹配。
_TOKEN_LIMIT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "context window",
    "prompt is too long",
    "too many tokens",
    "reduce the length",
    "exceeds the maximum",
    "max_tokens is too large",
)


def is_token_limit_error(text: str) -> bool:
    """判断错误文本是否表示输入超出上下文窗口。"""
    lowered = text.lower()
    return any(marker in lowered for marker in _TOKEN_LIMIT_MARKERS)


def parse_retry_after(value: str | None) -> float | None:
    """解析 HTTP ``Retry-After`` 的秒数或日期，输入异常时不抛错。"""
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
    return delay if delay >= 0 and isfinite(delay) else None
