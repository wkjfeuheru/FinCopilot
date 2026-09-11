"""Streaming Provider retry policy."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from finharness.provider.errors import ProviderError
from finharness.types import StreamChunk


@dataclass(frozen=True, slots=True)
class RetryPolicy:
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
    """Yield a stream, retrying only failures that occur before its first chunk."""
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
