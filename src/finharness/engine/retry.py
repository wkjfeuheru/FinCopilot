"""流式 Provider 的重试策略。"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from finharness.provider.errors import ProviderError
from finharness.types import StreamChunk


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """流式请求的重试策略参数。"""

    max_retries: int = 4
    base_delay_s: float = 1.0
    cap_delay_s: float = 30.0


async def stream_with_retry(
    stream_factory: Callable[[], AsyncIterator[StreamChunk]],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
    on_retry: Callable[[ProviderError, int, float], None] | None = None,
) -> AsyncIterator[StreamChunk]:
    """产出流，仅对首个 chunk 之前发生的失败进行重试。

    一旦已产出过 chunk（即响应已开始），失败将直接抛出，因为此时重试
    会重复或拼接已流出的内容。仅当错误标记为可重试、且重试次数未超出
    策略上限时才重试，退避延迟按指数增长并叠加抖动与错误自带的
    ``retry_after_s``。
    """
    retry_index = 0
    while True:
        emitted_chunk = False
        try:
            async for chunk in stream_factory():
                emitted_chunk = True
                yield chunk
            return
        except ProviderError as error:
            if emitted_chunk or not error.retryable or retry_index >= policy.max_retries:
                raise
            delay = (
                min(policy.cap_delay_s, policy.base_delay_s * 2**retry_index)
                + jitter(0, policy.base_delay_s)
                + (error.retry_after_s or 0.0)
            )
            if on_retry is not None:
                on_retry(error, retry_index, delay)
            retry_index += 1
            await sleep(delay)
