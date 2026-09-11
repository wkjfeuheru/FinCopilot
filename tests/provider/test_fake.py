import asyncio

from finharness.provider.fake import FakeProvider
from finharness.types import ModelUsage, Msg, StreamEvent


def test_fake_provider_emits_scripted_text():
    async def collect():
        provider = FakeProvider(["第一段", "第二段"])
        return [
            chunk
            async for chunk in provider.stream(
                system="", messages=[Msg.user("问题")], tools=[], usage=ModelUsage()
            )
        ]

    chunks = asyncio.run(collect())

    assert [chunk.event for chunk in chunks] == [
        StreamEvent.TEXT_DELTA,
        StreamEvent.TEXT_DELTA,
        StreamEvent.MESSAGE_END,
    ]
    assert "".join(chunk.data for chunk in chunks[:2]) == "第一段第二段"
