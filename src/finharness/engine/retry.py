"""流式 Provider 的重试策略。"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from finharness.provider.errors import ProviderError
from finharness.types import StreamChunk, StreamEvent


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
    """产出流，对首个 chunk 之前的失败、以及"作废整次输出"的失败进行重试。

    两条重试边界：

    * **首个 chunk 之前**的传输层失败——无内容流出，直接重放即可。
    * **流结束后**才发现的无效响应（``voids_output=True``，如工具参数被
      ``max_tokens`` 截断）——此时文本与工具片段已经流出，但问题出在响应整体，
      重放是安全的，前提是消费方先把它们丢掉。因此重试前先产出
      ``RESTART``，由消费方清空本轮累积。

    其余情况下，一旦流过 chunk 就不再重试：那会导致两次尝试的内容被拼接。
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
            if not error.retryable or retry_index >= policy.max_retries:
                raise
            if emitted_chunk and not error.voids_output:
                raise
            if error.voids_output:
                # 已流出的内容作废：先让消费方清空，再重放。
                yield StreamChunk(StreamEvent.RESTART)
            delay = (
                min(policy.cap_delay_s, policy.base_delay_s * 2**retry_index)
                + jitter(0, policy.base_delay_s)
                + (error.retry_after_s or 0.0)
            )
            if on_retry is not None:
                on_retry(error, retry_index, delay)
            retry_index += 1
            await sleep(delay)
