import asyncio

from finharness.engine.loop import AgentLoop
from finharness.provider.fake import FakeProvider
from finharness.types import AgentTurnOutcome, EngineEvent
from finharness.tools.registry import ToolRegistry


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


def test_agent_turn_outcome_has_reason_and_tool_call_defaults():
    outcome = AgentTurnOutcome(answer="ok")

    assert outcome.reason is None
    assert outcome.tool_calls == 0


def test_tool_registry_schemas_filters_names_in_registry_order():
    registry = ToolRegistry(data=None)

    schemas = registry.schemas({"get_indicators", "get_quote"})

    assert [schema["function"]["name"] for schema in schemas] == ["get_quote", "get_indicators"]


def test_tool_registry_read_only_marks_only_quote_kline_indicators():
    registry = ToolRegistry(data=None)

    assert registry.is_read_only("get_quote") is True
    assert registry.is_read_only("get_kline") is True
    assert registry.is_read_only("get_indicators") is True
    assert registry.is_read_only("unknown") is False
