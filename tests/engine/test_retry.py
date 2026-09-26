import asyncio

import pytest

from finharness.engine.retry import RetryPolicy, stream_with_retry
from finharness.provider.errors import (
    NetworkError,
    OutputTruncatedError,
    ProviderError,
    RateLimitError,
)
from finharness.types import StreamChunk, StreamEvent


def collect(iterator):
    async def run():
        return [chunk async for chunk in iterator]

    return asyncio.run(run())


def test_stream_with_retry_recovers_before_the_first_chunk():
    attempts = 0
    delays: list[float] = []
    retries: list[tuple[ProviderError, int, float]] = []

    async def attempt():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RateLimitError("limited", retry_after_s=2.0)
        yield StreamChunk(StreamEvent.MESSAGE_END)

    async def sleep(delay: float) -> None:
        delays.append(delay)

    chunks = collect(
        stream_with_retry(
            attempt,
            policy=RetryPolicy(max_retries=4, base_delay_s=1.0, cap_delay_s=30.0),
            sleep=sleep,
            jitter=lambda _low, _high: 0.5,
            on_retry=lambda error, index, delay: retries.append((error, index, delay)),
        )
    )

    assert [chunk.event for chunk in chunks] == [StreamEvent.MESSAGE_END]
    assert attempts == 3
    assert delays == [3.5, 4.5]
    assert [(index, delay) for _, index, delay in retries] == [(0, 3.5), (1, 4.5)]


def test_stream_with_retry_stops_after_configured_retry_budget():
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        raise RateLimitError("limited")
        yield StreamChunk(StreamEvent.MESSAGE_END)

    async def sleep(_delay: float) -> None:
        return None

    with pytest.raises(RateLimitError):
        collect(
            stream_with_retry(
                attempt,
                policy=RetryPolicy(),
                sleep=sleep,
                jitter=lambda _low, _high: 0.0,
            )
        )

    assert attempts == 5


def test_stream_with_retry_does_not_retry_after_any_chunk():
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        yield StreamChunk(StreamEvent.TEXT_DELTA, "partial")
        raise RateLimitError("limited")

    with pytest.raises(RateLimitError):
        collect(stream_with_retry(attempt, policy=RetryPolicy()))

    assert attempts == 1


def test_stream_with_retry_replays_when_the_error_voids_output():
    """流结束后才发现的无效响应（如被截断）应重放，并在重放前发出 RESTART。

    消费方据此丢弃上一次尝试的文本与工具片段；否则两次尝试会被拼接。
    """
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield StreamChunk(StreamEvent.TEXT_DELTA, "half a report")
            raise OutputTruncatedError("truncated at max_tokens")
        yield StreamChunk(StreamEvent.TEXT_DELTA, "full report")
        yield StreamChunk(StreamEvent.MESSAGE_END)

    async def sleep(_delay: float) -> None:
        return None

    chunks = collect(
        stream_with_retry(
            attempt,
            policy=RetryPolicy(),
            sleep=sleep,
            jitter=lambda _low, _high: 0.0,
        )
    )

    assert attempts == 2
    assert [chunk.event for chunk in chunks] == [
        StreamEvent.TEXT_DELTA,
        StreamEvent.RESTART,
        StreamEvent.TEXT_DELTA,
        StreamEvent.MESSAGE_END,
    ]
    # RESTART 之后的内容才是有效的，之前那份必须被消费方丢弃。
    assert [chunk.data for chunk in chunks if chunk.event is StreamEvent.TEXT_DELTA] == [
        "half a report",
        "full report",
    ]


def test_stream_with_retry_gives_up_after_the_replay_budget():
    """重放次数应受策略上限约束，而非无限重试。"""
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        yield StreamChunk(StreamEvent.TEXT_DELTA, "truncated")
        raise OutputTruncatedError("still truncated")

    async def sleep(_delay: float) -> None:
        return None

    with pytest.raises(OutputTruncatedError):
        collect(
            stream_with_retry(
                attempt,
                policy=RetryPolicy(max_retries=2),
                sleep=sleep,
                jitter=lambda _low, _high: 0.0,
            )
        )

    assert attempts == 3


@pytest.mark.parametrize(
    "error",
    [NetworkError("bad", retryable=False), ValueError("unexpected")],
)
def test_stream_with_retry_propagates_non_retryable_errors(error):
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        raise error
        yield StreamChunk(StreamEvent.MESSAGE_END)

    with pytest.raises(type(error)):
        collect(stream_with_retry(attempt, policy=RetryPolicy()))

    assert attempts == 1


def test_stream_with_retry_propagates_cancellation_without_retrying():
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError()
        yield StreamChunk(StreamEvent.MESSAGE_END)

    async def run():
        async for _chunk in stream_with_retry(attempt, policy=RetryPolicy()):
            pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert attempts == 1
