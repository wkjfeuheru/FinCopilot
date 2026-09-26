"""Provider 错误分类与重试元数据。"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite


class ProviderError(RuntimeError):
    """provider 失败，并携带供 engine 使用的重试信息。"""

    default_retryable = False

    # 重试前是否必须丢弃本次已流出的内容。传输层失败（连接被拒、首字节超时）
    # 发生在任何 chunk 之前，无内容可丢，故为 False；而"响应本身无效/被截断"
    # 是流**结束后**才发现的，此时文本与工具调用片段已经流出，重试必须先让
    # 消费方清空，否则两次尝试的内容会被拼接在一起。重试层据此决定是否
    # 向前发出 RESTART 信号。
    voids_output = False

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


class MalformedStreamError(NetworkError):
    """provider 流本身无效——分片 JSON 不完整、字段类型错误、SSE 载荷缺字段等。

    与 ``NetworkError`` 的差别在于**发现时机**：这一类总是在响应已经流出之后才
    被发现，因此已产出的文本/工具片段必须被视为作废。它默认可重试（同样的请求
    换一次采样往往就好了），且以 ``voids_output=True`` 通知重试层先重置消费方。
    """

    default_retryable = True
    voids_output = True


class OutputTruncatedError(MalformedStreamError):
    """模型输出触到 ``max_tokens`` 上限被截断（``finish_reason == "length"``）。

    这不是传输故障，而是预算不足：工具参数被截在半句 JSON 上、或正文缺少结尾。
    它与 ``MalformedStreamError`` 一样可重试且作废已流出内容，但单独成类是为了
    让"预算不足"在指标里可分辨——否则它会混进"网络错误"，运维看不到该调高
    ``max_tokens`` 的信号。
    """


class ToolArgumentsError(MalformedStreamError):
    """工具参数不是合法 JSON 对象——通常正是被截断的后果。

    在读取 ``finish_reason`` 之前它是唯一可见的证据；有了 ``finish_reason``
    之后应优先报 ``OutputTruncatedError``，以便区分"预算不足"与"模型吐了坏 JSON"。
    """


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
