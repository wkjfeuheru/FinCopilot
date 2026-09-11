"""Provider protocol and normalized message types."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from finharness.types import ModelUsage, Msg, StreamChunk


class Provider(ABC):
    @abstractmethod
    async def stream(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[dict],
        usage: ModelUsage,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError
