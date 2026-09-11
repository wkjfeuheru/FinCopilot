import asyncio

from finharness.engine.loop import AgentLoop
from finharness.provider.fake import FakeProvider
from finharness.types import EngineEvent


class Sink:
    def __init__(self):
        self.events: list[EngineEvent] = []

    async def emit(self, event: EngineEvent) -> None:
        self.events.append(event)


def test_agent_loop_emits_answer_and_done():
    async def run():
        sink = Sink()
        loop = AgentLoop(provider=FakeProvider(["hello", " world"]), output=sink)
        outcome = await loop.run("question")
        return outcome, sink.events, loop.messages

    outcome, events, messages = asyncio.run(run())

    assert outcome.answer == "hello world"
    assert [event.kind for event in events] == ["text_delta", "text_delta", "answer", "done"]
    assert [message.role for message in messages] == ["user", "assistant"]
