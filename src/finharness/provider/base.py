"""Provider 协议与规范化消息类型。"""

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
        """以流式方式执行一次补全请求，产出规范化 StreamChunk 事件。"""
        raise NotImplementedError
