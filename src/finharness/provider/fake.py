"""Explicit offline provider used by tests and local UI development."""

from collections.abc import AsyncIterator, Iterable

from finharness.provider.base import Provider
from finharness.types import ModelUsage, Msg, StreamChunk, StreamEvent


class FakeProvider(Provider):
    def __init__(self, chunks: Iterable[str], *, error: Exception | None = None):
        self.chunks = list(chunks)
        self.error = error
        self.requests: list[list[Msg]] = []

    async def stream(self, *, system: str, messages: list[Msg], tools: list[dict], usage: ModelUsage) -> AsyncIterator[StreamChunk]:
        self.requests.append(list(messages))
        if self.error:
            raise self.error
        for text in self.chunks:
            yield StreamChunk(StreamEvent.TEXT_DELTA, text)
        yield StreamChunk(StreamEvent.MESSAGE_END, ModelUsage(input_tokens=1, output_tokens=len(self.chunks)))
