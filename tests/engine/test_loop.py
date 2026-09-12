import asyncio
import json

import pytest

from finharness.config.settings import ContextSettings, Settings, ToolSettings
from finharness.engine.cost import SessionStats
from finharness.engine.loop import AgentLoop
from finharness.engine.retry import RetryPolicy
from finharness.provider.base import Provider
from finharness.provider.errors import RateLimitError
from finharness.provider.fake import FakeProvider
from finharness.tools.registry import ToolRegistry
from finharness.types import (
    AgentTurnOutcome,
    EngineEvent,
    ModelUsage,
    StreamChunk,
    StreamEvent,
    ToolResult,
    ToolUse,
)


class StepClock:
    """Deterministic clock advancing a fixed step per call."""

    def __init__(self, step_s: float = 0.25):
        self._value = 0.0
        self._step_s = step_s

    def __call__(self) -> float:
        value = self._value
        self._value += self._step_s
        return value


class Sink:
    def __init__(self):
        self.events: list[EngineEvent] = []

    async def emit(self, event: EngineEvent) -> None:
        self.events.append(event)


class StubRegistry:
    """Tool catalog double: the loop only needs names/resolve/schemas/read-only."""

    def __init__(self, tools: dict[str, object] | None = None, *, read_only: set[str] | None = None):
        self.tools = dict(tools or {})
        self.read_only = set(self.tools) if read_only is None else set(read_only)

    def names(self) -> list[str]:
        return list(self.tools)

    def resolve(self, name: str):
        return self.tools.get(name)

    def schemas(self, names: set[str] | None = None) -> list[dict]:
        return [
            {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
            for name in self.tools
            if names is None or name in names
        ]

    def is_read_only(self, name: str) -> bool:
        return name in self.read_only


class ScriptedProvider(Provider):
    """Replays canned rounds and records every request the loop makes."""

    def __init__(self, rounds: list[list[StreamChunk]] | None = None, *, error: Exception | None = None):
        self.rounds = list(rounds or [])
        self.error = error
        self.requests: list[dict] = []

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
        self.requests.append(
            {
                "system": system,
                "roles": [message.role for message in messages],
                "messages": list(messages),
                "tools": list(tools),
            }
        )
        if self.error is not None:
            raise self.error
        script = self.rounds.pop(0) if self.rounds else [message_end()]
        if isinstance(script, Exception):
            raise script
        for chunk in script:
            yield chunk


class ChunkThenErrorProvider(Provider):
    """Yields one delta, then fails mid-stream so retry must not trigger."""

    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
        self.calls += 1
        yield StreamChunk(StreamEvent.TEXT_DELTA, "partial")
        raise self.error


class RecordingTool:
    """Read-only tool double that records arguments and can fail, stall or raise."""

    def __init__(
        self,
        name: str,
        *,
        content: str = "content",
        ok: bool = True,
        error: str | None = None,
        exc: Exception | None = None,
        delay: float = 0.0,
    ):
        self.name = name
        self.content = content
        self.ok = ok
        self.error = error
        self.exc = exc
        self.delay = delay
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return ToolResult(content=self.content, ok=self.ok, error=self.error)


class ParallelGate:
    """Releases only once every participant arrives, so serial execution times out."""

    def __init__(self, participants: int, *, timeout_s: float = 1.0):
        self.remaining = participants
        self.released = asyncio.Event()
        self.timeout_s = timeout_s

    async def wait(self) -> None:
        self.remaining -= 1
        if self.remaining <= 0:
            self.released.set()
        await asyncio.wait_for(self.released.wait(), self.timeout_s)


class GatedTool(RecordingTool):
    """Records when it starts and finishes so call order can be observed."""

    def __init__(self, name: str, gate: ParallelGate, *, content: str, finish_delay: float = 0.0):
        super().__init__(name, content=content)
        self.gate = gate
        self.finish_delay = finish_delay
        self.finished = False

    async def run(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        await self.gate.wait()
        if self.finish_delay:
            await asyncio.sleep(self.finish_delay)
        self.finished = True
        return ToolResult(content=self.content)


class CancellableTool:
    """Signals when it starts and records whether it observed cancellation."""

    def __init__(self, name: str):
        self.name = name
        self.started = asyncio.Event()
        self.cancelled = False

    async def run(self, **kwargs) -> ToolResult:
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ToolResult(content="never returned")


def message_end(*tool_uses: ToolUse, input_tokens: int = 1, output_tokens: int = 1) -> StreamChunk:
    return StreamChunk(
        StreamEvent.MESSAGE_END,
        ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens, tool_uses=list(tool_uses)),
    )


def text_round(*parts: str, input_tokens: int = 1, output_tokens: int = 1) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in parts] + [
        message_end(input_tokens=input_tokens, output_tokens=output_tokens)
    ]


def tool_round(*tool_uses: ToolUse, draft: tuple[str, ...] = (), input_tokens: int = 1, output_tokens: int = 1) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in draft] + [
        message_end(*tool_uses, input_tokens=input_tokens, output_tokens=output_tokens)
    ]


def make_settings(
    *,
    max_turns: int = 30,
    max_result_tokens: int = 1000,
    timeout_default_s: int = 30,
    timeout_overrides: dict[str, int] | None = None,
) -> Settings:
    return Settings(
        context=ContextSettings(max_turns=max_turns, max_result_tokens=max_result_tokens),
        tools=ToolSettings(timeout_default_s=timeout_default_s, timeout_overrides=timeout_overrides or {}),
    )


def make_loop(
    provider: Provider,
    *,
    registry=None,
    settings: Settings | None = None,
    output=None,
    system: str = "test-system",
    retry_policy: RetryPolicy | None = None,
    stats: SessionStats | None = None,
) -> AgentLoop:
    return AgentLoop(
        provider=provider,
        registry=registry if registry is not None else StubRegistry(),
        settings=settings or make_settings(),
        system=system,
        output=output,
        retry_policy=retry_policy,
        stats=stats,
    )


def kinds(events: list[EngineEvent]) -> list[str]:
    return [event.kind for event in events]


def test_agent_turn_outcome_has_reason_and_tool_call_defaults():
    outcome = AgentTurnOutcome(answer="ok")

    assert outcome.reason is None
    assert outcome.tool_calls == 0
    assert outcome.retry_count == 0
    assert outcome.tool_duration_ms == 0


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


def test_final_turn_emits_deltas_then_answer_then_done():
    async def run():
        sink = Sink()
        provider = ScriptedProvider([text_round("hello", " world", input_tokens=1, output_tokens=2)])
        loop = make_loop(provider, output=sink)
        outcome = await loop.run("question")
        return outcome, sink.events, loop.messages, provider

    outcome, events, messages, provider = asyncio.run(run())

    assert kinds(events) == ["text_delta", "text_delta", "answer", "done"]
    assert [event.data["text"] for event in events[:2]] == ["hello", " world"]
    assert events[2].data == {"text": "hello world"}
    done = events[3].data
    assert done["succeeded"] is True
    assert done["reason"] is None
    assert done["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert done["tool_calls"] == 0
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].content == "hello world"
    assert outcome.answer == "hello world"
    assert outcome.succeeded is True
    assert outcome.reason is None
    assert outcome.tool_calls == 0
    assert outcome.usage.input_tokens == 1
    assert outcome.usage.output_tokens == 2
    assert provider.requests[0]["system"] == "test-system"


def test_agent_loop_uses_registry_schemas_and_injected_system_prompt():
    async def run():
        provider = ScriptedProvider([text_round("done")])
        registry = StubRegistry({"get_quote": object(), "get_kline": object()})
        loop = make_loop(provider, registry=registry, system="system-prompt")
        await loop.run("question")
        return provider

    provider = asyncio.run(run())

    request = provider.requests[0]
    assert request["system"] == "system-prompt"
    assert [schema["function"]["name"] for schema in request["tools"]] == ["get_quote", "get_kline"]


def test_agent_loop_works_with_the_offline_fake_provider():
    async def run():
        sink = Sink()
        loop = make_loop(FakeProvider(["hello", " world"]), output=sink)
        outcome = await loop.run("question")
        return outcome, sink.events

    outcome, events = asyncio.run(run())

    assert outcome.answer == "hello world"
    assert kinds(events) == ["text_delta", "text_delta", "answer", "done"]


def test_tool_round_hides_draft_and_backfills_one_structured_result_per_call():
    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价 600519 100.0")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [
                tool_round(
                    ToolUse("call_1", "get_quote", {"symbol": "600519"}),
                    draft=("草稿：", "正在查询"),
                ),
                text_round("正式答案"),
            ]
        )
        loop = make_loop(provider, registry=registry, output=sink)
        outcome = await loop.run("贵州茅台多少钱")
        return outcome, sink.events, loop.messages, provider, tool

    outcome, events, messages, provider, tool = asyncio.run(run())

    streamed_text = [event.data["text"] for event in events if event.kind == "text_delta"]
    answered_text = [event.data["text"] for event in events if event.kind == "answer"]
    assert "草稿" not in "".join(streamed_text)
    assert streamed_text == ["正式答案"]
    assert answered_text == ["正式答案"]
    assert [message.role for message in messages] == ["user", "assistant", "tool_result", "assistant"]
    assert messages[1].content is None
    assert messages[1].tool_uses[0].call_id == "call_1"
    assert messages[1].tool_uses[0].name == "get_quote"
    assert [call_id for call_id, _ in messages[2].tool_results] == ["call_1"]
    payload = json.loads(messages[2].tool_results[0][1])
    assert payload == {"ok": True, "content": "报价 600519 100.0", "error": None}
    assert tool.calls == [{"symbol": "600519"}]
    assert kinds(events) == ["tool_status", "tool_status", "text_delta", "answer", "done"]
    assert [event.data["status"] for event in events[:2]] == ["started", "completed"]
    assert events[1].data["call_id"] == "call_1"
    assert provider.requests[1]["roles"] == ["user", "assistant", "tool_result"]
    assert outcome.answer == "正式答案"
    assert outcome.tool_calls == 1
    assert outcome.succeeded is True
    assert events[-1].data["tool_calls"] == 1


def test_read_only_tools_run_concurrently_and_backfill_in_tool_use_order():
    async def run():
        gate = ParallelGate(2)
        quote = GatedTool("get_quote", gate, content="报价", finish_delay=0.05)
        kline = GatedTool("get_kline", gate, content="K线")
        registry = StubRegistry({"get_quote": quote, "get_kline": kline})
        provider = ScriptedProvider(
            [
                tool_round(
                    ToolUse("call_quote", "get_quote", {"symbol": "600519"}),
                    ToolUse("call_kline", "get_kline", {"symbol": "600519"}),
                ),
                text_round("both done"),
            ]
        )
        loop = make_loop(provider, registry=registry)
        outcome = await loop.run("对比报价与K线")
        return outcome, loop.messages, quote, kline

    outcome, messages, quote, kline = asyncio.run(run())

    assert quote.finished is True
    assert kline.finished is True
    assert [call_id for call_id, _ in messages[2].tool_results] == ["call_quote", "call_kline"]
    assert [json.loads(raw)["content"] for _, raw in messages[2].tool_results] == ["报价", "K线"]
    assert outcome.tool_calls == 2
    assert outcome.answer == "both done"


def test_unknown_and_non_read_only_tools_backfill_failures_and_keep_going():
    async def run():
        sink = Sink()
        writer = RecordingTool("write_note")
        reader = RecordingTool("get_quote", content="报价")
        registry = StubRegistry({"get_quote": reader, "write_note": writer}, read_only={"get_quote"})
        provider = ScriptedProvider(
            [
                tool_round(
                    ToolUse("call_unknown", "missing_tool", {}),
                    ToolUse("call_write", "write_note", {"text": "x"}),
                    ToolUse("call_read", "get_quote", {"symbol": "600519"}),
                ),
                text_round("final"),
            ]
        )
        loop = make_loop(provider, registry=registry, output=sink)
        outcome = await loop.run("写一条笔记并读取报价")
        return outcome, sink.events, loop.messages, writer, reader

    outcome, events, messages, writer, reader = asyncio.run(run())

    payloads = {call_id: json.loads(raw) for call_id, raw in messages[2].tool_results}
    assert [call_id for call_id, _ in messages[2].tool_results] == ["call_unknown", "call_write", "call_read"]
    assert payloads["call_unknown"]["ok"] is False
    assert "unknown tool" in payloads["call_unknown"]["error"]
    assert payloads["call_write"]["ok"] is False
    assert "read-only" in payloads["call_write"]["error"]
    assert payloads["call_read"] == {"ok": True, "content": "报价", "error": None}
    assert writer.calls == []
    assert reader.calls == [{"symbol": "600519"}]
    failed = [event for event in events if event.kind == "tool_status" and event.data["status"] == "failed"]
    assert [event.data["call_id"] for event in failed] == ["call_unknown", "call_write"]
    assert outcome.answer == "final"
    assert outcome.succeeded is True
    assert outcome.tool_calls == 3


def test_tool_exception_and_timeout_backfill_failures_and_model_recovers():
    async def run():
        sink = Sink()
        boom = RecordingTool("boom", exc=RuntimeError("adapter exploded"))
        slow = RecordingTool("slow", delay=5.0)
        registry = StubRegistry({"boom": boom, "slow": slow}, read_only={"boom", "slow"})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("call_boom", "boom", {}), ToolUse("call_slow", "slow", {})),
                text_round("recovered"),
            ]
        )
        loop = make_loop(provider, registry=registry, settings=make_settings(timeout_default_s=1), output=sink)
        outcome = await loop.run("调用两个坏工具")
        return outcome, sink.events, loop.messages

    outcome, events, messages = asyncio.run(run())

    payloads = {call_id: json.loads(raw) for call_id, raw in messages[2].tool_results}
    assert payloads["call_boom"]["ok"] is False
    assert "adapter exploded" in payloads["call_boom"]["error"]
    assert payloads["call_slow"]["ok"] is False
    assert "timeout" in payloads["call_slow"]["error"]
    assert [message.role for message in messages] == ["user", "assistant", "tool_result", "assistant"]
    failed = [event for event in events if event.kind == "tool_status" and event.data["status"] == "failed"]
    assert [event.data["call_id"] for event in failed] == ["call_boom", "call_slow"]
    assert outcome.answer == "recovered"
    assert outcome.succeeded is True
    assert outcome.tool_calls == 2


def test_tool_result_content_is_truncated_to_the_configured_character_limit():
    async def run():
        tool = RecordingTool("get_kline", content="x" * 100)
        registry = StubRegistry({"get_kline": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("call_1", "get_kline", {})), text_round("ok")]
        )
        loop = make_loop(provider, registry=registry, settings=make_settings(max_result_tokens=40))
        await loop.run("要很长的K线")
        return loop.messages, provider

    messages, provider = asyncio.run(run())

    content = json.loads(messages[2].tool_results[0][1])["content"]
    assert len(content) == 40
    assert content == "x" * 28 + "\n[truncated]"
    sent_to_model = json.loads(provider.requests[1]["messages"][2].tool_results[0][1])
    assert sent_to_model["content"] == content
    assert "x" * 100 not in json.dumps(provider.requests[1]["messages"][2].tool_results)


def test_truncation_keeps_only_the_prefix_when_the_limit_is_shorter_than_the_marker():
    async def run():
        tool = RecordingTool("get_kline", content="abcdefghij")
        registry = StubRegistry({"get_kline": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("call_1", "get_kline", {})), text_round("ok")]
        )
        loop = make_loop(provider, registry=registry, settings=make_settings(max_result_tokens=5))
        await loop.run("短上限")
        return loop.messages

    messages = asyncio.run(run())

    assert json.loads(messages[2].tool_results[0][1])["content"] == "abcde"


def test_short_tool_result_is_backfilled_unchanged():
    async def run():
        tool = RecordingTool("get_quote", content="报价")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("call_1", "get_quote", {})), text_round("ok")]
        )
        loop = make_loop(provider, registry=registry, settings=make_settings(max_result_tokens=1000))
        await loop.run("短结果")
        return loop.messages

    messages = asyncio.run(run())

    assert json.loads(messages[2].tool_results[0][1])["content"] == "报价"


def test_exhausted_tool_turns_report_reason_and_emit_one_error_and_one_done():
    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("call_1", "get_quote", {})),
                tool_round(ToolUse("call_2", "get_quote", {})),
            ]
        )
        loop = make_loop(provider, registry=registry, settings=make_settings(max_turns=2), output=sink)
        outcome = await loop.run("一直调用工具")
        return outcome, sink.events, loop.messages, provider

    outcome, events, messages, provider = asyncio.run(run())

    assert outcome.succeeded is False
    assert outcome.reason == "max_turns_exhausted"
    assert outcome.answer == ""
    assert outcome.tool_calls == 2
    assert outcome.usage.input_tokens == 2
    assert kinds(events).count("error") == 1
    assert kinds(events).count("done") == 1
    assert kinds(events)[-2:] == ["error", "done"]
    assert events[-2].data["reason"] == "max_turns_exhausted"
    assert events[-1].data["succeeded"] is False
    assert events[-1].data["reason"] == "max_turns_exhausted"
    assert events[-1].data["tool_calls"] == 2
    assert "answer" not in kinds(events)
    assert len(provider.requests) == 2
    assert [message.role for message in messages] == ["user", "assistant", "tool_result", "assistant", "tool_result"]


def test_provider_error_reports_reason_and_emits_one_error_and_one_done():
    async def run():
        sink = Sink()
        provider = ScriptedProvider(error=RuntimeError("provider down"))
        loop = make_loop(provider, output=sink)
        outcome = await loop.run("question")
        return outcome, sink.events

    outcome, events = asyncio.run(run())

    assert outcome.succeeded is False
    assert outcome.reason == "provider_error"
    assert outcome.answer == ""
    assert outcome.error == "provider down"
    assert outcome.tool_calls == 0
    assert kinds(events) == ["error", "done"]
    assert events[0].data == {
        "kind": "RuntimeError",
        "message": "provider down",
        "reason": "provider_error",
    }
    assert events[1].data["succeeded"] is False
    assert events[1].data["reason"] == "provider_error"
    assert events[1].data["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_provider_error_after_a_tool_round_keeps_the_tool_call_count():
    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("call_1", "get_quote", {})), RuntimeError("provider down")]
        )
        loop = make_loop(provider, registry=registry, output=sink)
        outcome = await loop.run("先查报价")
        return outcome, sink.events, loop.messages

    outcome, events, messages = asyncio.run(run())

    assert outcome.reason == "provider_error"
    assert outcome.tool_calls == 1
    assert kinds(events) == ["tool_status", "tool_status", "error", "done"]
    assert events[-1].data["tool_calls"] == 1
    assert events[-1].data["usage"] == {"input_tokens": 1, "output_tokens": 1}
    assert [message.role for message in messages] == ["user", "assistant", "tool_result"]


def test_cancelling_run_cancels_running_tools_and_never_emits_done():
    async def run():
        sink = Sink()
        tool = CancellableTool("get_quote")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider([tool_round(ToolUse("call_1", "get_quote", {}))])
        loop = make_loop(provider, registry=registry, output=sink)
        task = asyncio.create_task(loop.run("查询报价"))
        await asyncio.wait_for(tool.started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return sink.events, tool

    events, tool = asyncio.run(run())

    assert tool.cancelled is True
    assert kinds(events) == ["tool_status"]
    assert events[0].data["status"] == "started"
    assert "done" not in kinds(events)
    assert "error" not in kinds(events)


def test_cancelled_tool_round_backfills_a_failure_for_every_call_id():
    async def run():
        sink = Sink()
        tool = CancellableTool("get_quote")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("call_1", "get_quote", {}), ToolUse("call_2", "get_quote", {})),
                text_round("第二轮答案"),
            ]
        )
        loop = make_loop(provider, registry=registry, output=sink)
        task = asyncio.create_task(loop.run("查询报价"))
        await asyncio.wait_for(tool.started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        outcome = await loop.run("再问一次")
        return loop.messages, provider, outcome, sink.events

    messages, provider, outcome, events = asyncio.run(run())

    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool_result",
        "user",
        "assistant",
    ]
    payloads = [json.loads(raw) for _, raw in messages[2].tool_results]
    assert [call_id for call_id, _ in messages[2].tool_results] == ["call_1", "call_2"]
    assert [payload["ok"] for payload in payloads] == [False, False]
    assert all("cancel" in payload["error"] for payload in payloads)
    second_request = provider.requests[1]["messages"]
    requested = {tool_use.call_id for message in second_request for tool_use in message.tool_uses}
    answered = {
        call_id
        for message in second_request
        if message.role == "tool_result"
        for call_id, _ in message.tool_results
    }
    assert requested == answered == {"call_1", "call_2"}
    assert outcome.answer == "第二轮答案"
    assert kinds(events)[-1] == "done"


def test_provider_retry_recovers_before_the_first_chunk_and_counts_retries():
    async def run():
        sink = Sink()
        provider = ScriptedProvider(
            [RateLimitError("limited"), RateLimitError("limited"), text_round("ok")]
        )
        loop = make_loop(
            provider,
            output=sink,
            retry_policy=RetryPolicy(
                max_retries=4, base_delay_s=0.0, cap_delay_s=0.0
            ),
        )
        outcome = await loop.run("question")
        return outcome, sink.events, provider, loop

    outcome, events, provider, loop = asyncio.run(run())

    assert outcome.succeeded is True
    assert outcome.retry_count == 2
    assert outcome.tool_calls == 0
    assert outcome.usage.input_tokens == 1
    assert len(provider.requests) == 3
    assert loop.stats.retry_count == 2
    done = events[-1].data
    assert done["retry_count"] == 2
    assert done["tool_duration_ms"] == 0
    assert kinds(events) == ["text_delta", "answer", "done"]


def test_provider_retry_exhaustion_fails_with_retry_count_in_done():
    async def run():
        sink = Sink()
        provider = ScriptedProvider(
            [RateLimitError("limited"), RateLimitError("limited"),
             RateLimitError("limited"), RateLimitError("limited"),
             RateLimitError("limited")]
        )
        loop = make_loop(
            provider,
            output=sink,
            retry_policy=RetryPolicy(
                max_retries=4, base_delay_s=0.0, cap_delay_s=0.0
            ),
        )
        outcome = await loop.run("question")
        return outcome, sink.events, provider

    outcome, events, provider = asyncio.run(run())

    assert outcome.succeeded is False
    assert outcome.reason == "provider_error"
    assert outcome.retry_count == 4
    assert len(provider.requests) == 5
    assert kinds(events) == ["error", "done"]
    assert events[-1].data["retry_count"] == 4
    assert events[-1].data["tool_duration_ms"] == 0


def test_provider_failure_after_a_chunk_is_not_retried():
    async def run():
        sink = Sink()
        provider = ChunkThenErrorProvider(RateLimitError("mid-stream failure"))
        loop = make_loop(
            provider,
            output=sink,
            retry_policy=RetryPolicy(
                max_retries=4, base_delay_s=0.0, cap_delay_s=0.0
            ),
        )
        outcome = await loop.run("question")
        return outcome, sink.events, provider

    outcome, events, provider = asyncio.run(run())

    assert outcome.succeeded is False
    assert outcome.reason == "provider_error"
    assert provider.calls == 1
    assert outcome.retry_count == 0
    assert events[-1].data["retry_count"] == 0


def test_tool_timing_and_accumulated_stats_reach_the_done_event():
    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("call_1", "get_quote", {"symbol": "600519"})),
                text_round("ok"),
            ]
        )
        stats = SessionStats(clock=StepClock())
        loop = make_loop(
            provider, registry=registry, output=sink, stats=stats
        )
        outcome = await loop.run("查询报价")
        return outcome, sink.events, loop

    outcome, events, loop = asyncio.run(run())

    statuses = [event.data for event in events if event.kind == "tool_status"]
    assert statuses[0]["status"] == "started"
    assert statuses[1]["status"] == "completed"
    assert statuses[1]["duration_ms"] == 250
    snapshot = loop.stats.snapshot()
    assert snapshot.tool_calls == 1
    assert snapshot.tool_duration_ms == 250
    assert dict(snapshot.per_tool) == {"get_quote": {"count": 1, "duration_ms": 250}}
    assert outcome.tool_duration_ms == 250
    done = events[-1].data
    assert done["tool_calls"] == 1
    assert done["retry_count"] == 0
    assert done["tool_duration_ms"] == 250


def test_unknown_and_denied_tools_count_as_requests_without_duration():
    async def run():
        sink = Sink()
        registry = StubRegistry({"get_quote": RecordingTool("get_quote")}, read_only=set())
        provider = ScriptedProvider(
            [
                tool_round(
                    ToolUse("call_unknown", "missing_tool", {}),
                    ToolUse("call_read", "get_quote", {}),
                ),
                text_round("ok"),
            ]
        )
        stats = SessionStats(clock=StepClock())
        loop = make_loop(provider, registry=registry, output=sink, stats=stats)
        outcome = await loop.run("混合调用")
        return outcome, sink.events, loop

    outcome, events, loop = asyncio.run(run())

    snapshot = loop.stats.snapshot()
    assert snapshot.tool_calls == 2
    assert snapshot.tool_duration_ms == 0
    assert dict(snapshot.per_tool) == {
        "missing_tool": {"count": 1, "duration_ms": 0},
        "get_quote": {"count": 1, "duration_ms": 0},
    }
    assert outcome.tool_duration_ms == 0
    failed = [event.data for event in events if event.kind == "tool_status" and event.data["status"] == "failed"]
    assert all("duration_ms" not in data for data in failed)


def test_failed_tool_result_reports_failed_status_with_duration():
    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="stale", ok=False, error="denied by source")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("call_1", "get_quote", {})),
                text_round("ok"),
            ]
        )
        stats = SessionStats(clock=StepClock())
        loop = make_loop(provider, registry=registry, output=sink, stats=stats)
        outcome = await loop.run("查询报价")
        return outcome, sink.events, loop

    outcome, events, loop = asyncio.run(run())

    completed = [event.data for event in events if event.kind == "tool_status" and event.data["status"] != "started"]
    assert completed[0]["status"] == "failed"
    assert completed[0]["ok"] is False
    assert completed[0]["duration_ms"] == 250
    assert loop.stats.snapshot().tool_duration_ms == 250
    assert outcome.tool_duration_ms == 250
