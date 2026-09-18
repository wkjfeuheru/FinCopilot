import asyncio
import json

import pytest

from finharness.config.settings import ContextSettings, Settings, ToolSettings
from finharness.engine.cost import SessionStats
from finharness.engine.loop import AgentLoop
from finharness.engine.retry import RetryPolicy
from finharness.tools.base import PermissionLevel
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
    """确定性时钟，每次调用前进固定步长。"""

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
    """工具目录替身：loop 只需要 names/resolve/schemas/read-only。"""

    def __init__(
        self,
        tools: dict[str, object] | None = None,
        *,
        read_only: set[str] | None = None,
    ):
        self.tools = dict(tools or {})
        self.read_only = set(self.tools) if read_only is None else set(read_only)

    def names(self) -> list[str]:
        return list(self.tools)

    def resolve(self, name: str):
        return self.tools.get(name)

    def schemas(self, names: set[str] | None = None) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": name, "description": "", "parameters": {}},
            }
            for name in self.tools
            if names is None or name in names
        ]

    def is_read_only(self, name: str) -> bool:
        return name in self.read_only


class ScriptedProvider(Provider):
    """重放预置的 round，并记录 loop 发出的每一个请求。"""

    def __init__(
        self,
        rounds: list[list[StreamChunk]] | None = None,
        *,
        error: Exception | None = None,
    ):
        self.rounds = list(rounds or [])
        self.error = error
        self.requests: list[dict] = []

    async def stream(
        self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage
    ):
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
    """先产出一个 delta，然后在流中途失败，因此不应触发重试。"""

    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    async def stream(
        self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage
    ):
        self.calls += 1
        yield StreamChunk(StreamEvent.TEXT_DELTA, "partial")
        raise self.error


class RecordingTool:
    """只读工具替身，记录参数，并可以失败、停滞或抛出异常。"""

    permission = PermissionLevel.READ
    timeout = None  # 继承 settings.tools.timeout_default_s

    def __init__(
        self,
        name: str,
        *,
        content: str = "content",
        ok: bool = True,
        error: str | None = None,
        exc: Exception | None = None,
        delay: float = 0.0,
        permission: PermissionLevel = PermissionLevel.READ,
    ):
        self.name = name
        self.content = content
        self.ok = ok
        self.error = error
        self.exc = exc
        self.delay = delay
        self.permission = permission
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return ToolResult(content=self.content, ok=self.ok, error=self.error)


class ParallelGate:
    """只有在所有参与者都到达后才放行，因此串行执行会超时。"""

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
    """记录开始与结束的时刻，以便观察调用顺序。"""

    def __init__(
        self, name: str, gate: ParallelGate, *, content: str, finish_delay: float = 0.0
    ):
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
    """在开始时发出信号，并记录它是否观察到了取消。"""

    permission = PermissionLevel.READ
    timeout = None

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


def message_end(
    *tool_uses: ToolUse, input_tokens: int = 1, output_tokens: int = 1
) -> StreamChunk:
    return StreamChunk(
        StreamEvent.MESSAGE_END,
        ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_uses=list(tool_uses),
        ),
    )


def text_round(
    *parts: str, input_tokens: int = 1, output_tokens: int = 1
) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in parts] + [
        message_end(input_tokens=input_tokens, output_tokens=output_tokens)
    ]


def tool_round(
    *tool_uses: ToolUse,
    draft: tuple[str, ...] = (),
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> list[StreamChunk]:
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
        context=ContextSettings(
            max_turns=max_turns, max_result_tokens=max_result_tokens
        ),
        tools=ToolSettings(
            timeout_default_s=timeout_default_s,
            timeout_overrides=timeout_overrides or {},
        ),
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
    coordinator=None,
    observer=None,
) -> AgentLoop:
    return AgentLoop(
        provider=provider,
        registry=registry if registry is not None else StubRegistry(),
        settings=settings or make_settings(),
        system=system,
        output=output,
        retry_policy=retry_policy,
        stats=stats,
        coordinator=coordinator,
        observer=observer,
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

    assert [schema["function"]["name"] for schema in schemas] == [
        "get_quote",
        "get_indicators",
    ]


def test_tool_registry_read_only_marks_only_quote_kline_indicators():
    registry = ToolRegistry(data=None)

    assert registry.is_read_only("get_quote") is True
    assert registry.is_read_only("get_kline") is True
    assert registry.is_read_only("get_indicators") is True
    assert registry.is_read_only("unknown") is False


def test_final_turn_emits_deltas_then_answer_then_done():
    async def run():
        sink = Sink()
        provider = ScriptedProvider(
            [text_round("hello", " world", input_tokens=1, output_tokens=2)]
        )
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
    assert done["usage"]["input_tokens"] == 1
    assert done["usage"]["output_tokens"] == 2
    # cache 拆分项始终存在；为 0 表示 provider 未上报。
    assert done["usage"]["cache_hit_tokens"] == 0
    assert done["usage"]["cache_miss_tokens"] == 0
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
    assert [schema["function"]["name"] for schema in request["tools"]] == [
        "get_quote",
        "get_kline",
    ]


def test_agent_loop_works_with_the_offline_fake_provider():
    async def run():
        sink = Sink()
        loop = make_loop(FakeProvider(["hello", " world"]), output=sink)
        outcome = await loop.run("question")
        return outcome, sink.events

    outcome, events = asyncio.run(run())

    assert outcome.answer == "hello world"
    assert kinds(events) == ["text_delta", "text_delta", "answer", "done"]


def test_tool_round_streams_draft_then_resets_it_before_the_final_answer():
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

    # 文本在生成时即流式输出，因此草稿会一直可见，直到该 round
    # 被判定为 tool call；此时 text_reset 会清除它，只有最终
    # 答案会在下一个 round 流式输出。
    streamed_text = [
        event.data["text"] for event in events if event.kind == "text_delta"
    ]
    answered_text = [event.data["text"] for event in events if event.kind == "answer"]
    assert streamed_text == ["草稿：", "正在查询", "正式答案"]
    assert answered_text == ["正式答案"]
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]
    assert messages[1].content is None
    assert messages[1].tool_uses[0].call_id == "call_1"
    assert messages[1].tool_uses[0].name == "get_quote"
    assert [call_id for call_id, _ in messages[2].tool_results] == ["call_1"]
    payload = json.loads(messages[2].tool_results[0][1])
    assert payload == {"ok": True, "content": "报价 600519 100.0", "error": None}
    assert tool.calls == [{"symbol": "600519"}]
    assert kinds(events) == [
        "text_delta",
        "text_delta",
        "text_reset",
        "tool_status",
        "tool_status",
        "text_delta",
        "answer",
        "done",
    ]
    assert [event.data["status"] for event in events[3:5]] == ["started", "completed"]
    assert events[4].data["call_id"] == "call_1"
    # transcript 前缀是真实历史；末尾的 'user' 是仅为本次请求追加的
    # 研究状态视图 (docs 3.3)。
    roles = provider.requests[1]["roles"]
    assert roles[:3] == ["user", "assistant", "tool_result"]
    assert all(role == "user" for role in roles[3:])
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
    assert [call_id for call_id, _ in messages[2].tool_results] == [
        "call_quote",
        "call_kline",
    ]
    assert [json.loads(raw)["content"] for _, raw in messages[2].tool_results] == [
        "报价",
        "K线",
    ]
    assert outcome.tool_calls == 2
    assert outcome.answer == "both done"


def test_unknown_and_non_read_only_tools_backfill_failures_and_keep_going():
    async def run():
        sink = Sink()
        writer = RecordingTool("write_note", permission=PermissionLevel.WRITE)
        reader = RecordingTool("get_quote", content="报价")
        registry = StubRegistry(
            {"get_quote": reader, "write_note": writer}, read_only={"get_quote"}
        )
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
    assert [call_id for call_id, _ in messages[2].tool_results] == [
        "call_unknown",
        "call_write",
        "call_read",
    ]
    assert payloads["call_unknown"]["ok"] is False
    assert "unknown tool" in payloads["call_unknown"]["error"]
    assert payloads["call_write"]["ok"] is False
    assert "read-only" in payloads["call_write"]["error"]
    assert payloads["call_read"] == {"ok": True, "content": "报价", "error": None}
    assert writer.calls == []
    assert reader.calls == [{"symbol": "600519"}]
    failed = [
        event
        for event in events
        if event.kind == "tool_status" and event.data["status"] == "failed"
    ]
    assert [event.data["call_id"] for event in failed] == ["call_unknown", "call_write"]
    assert outcome.answer == "final"
    assert outcome.succeeded is True
    assert outcome.tool_calls == 3


def test_tool_exception_and_timeout_backfill_failures_and_model_recovers():
    async def run():
        sink = Sink()
        boom = RecordingTool("boom", exc=RuntimeError("adapter exploded"))
        slow = RecordingTool("slow", delay=5.0)
        registry = StubRegistry(
            {"boom": boom, "slow": slow}, read_only={"boom", "slow"}
        )
        provider = ScriptedProvider(
            [
                tool_round(
                    ToolUse("call_boom", "boom", {}), ToolUse("call_slow", "slow", {})
                ),
                text_round("recovered"),
            ]
        )
        loop = make_loop(
            provider,
            registry=registry,
            settings=make_settings(timeout_default_s=1),
            output=sink,
        )
        outcome = await loop.run("调用两个坏工具")
        return outcome, sink.events, loop.messages

    outcome, events, messages = asyncio.run(run())

    payloads = {call_id: json.loads(raw) for call_id, raw in messages[2].tool_results}
    assert payloads["call_boom"]["ok"] is False
    assert "adapter exploded" in payloads["call_boom"]["error"]
    assert payloads["call_slow"]["ok"] is False
    assert "timeout" in payloads["call_slow"]["error"]
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]
    failed = [
        event
        for event in events
        if event.kind == "tool_status" and event.data["status"] == "failed"
    ]
    assert [event.data["call_id"] for event in failed] == ["call_boom", "call_slow"]
    assert outcome.answer == "recovered"
    assert outcome.succeeded is True
    assert outcome.tool_calls == 2


def test_tool_result_content_is_truncated_to_the_configured_token_limit():
    """max_result_tokens 是 token 预算，因此中文会按真实数量被截断。"""

    async def run():
        tool = RecordingTool("get_kline", content="茅" * 400)
        registry = StubRegistry({"get_kline": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("call_1", "get_kline", {})), text_round("ok")]
        )
        loop = make_loop(
            provider, registry=registry, settings=make_settings(max_result_tokens=40)
        )
        await loop.run("要很长的K线")
        return loop.messages, provider, loop

    messages, provider, loop = asyncio.run(run())

    content = json.loads(messages[2].tool_results[0][1])["content"]
    # 截断标记会追加在预算内的前缀之后，并说明被省略的规模。
    assert loop.TRUNCATION_MARKER in content
    assert "已省略约" in content
    body = content.split(loop.TRUNCATION_MARKER, 1)[0]
    assert loop.memory.counter.count(body).tokens <= 40
    # 未被截断的 payload 绝不能已被转发。
    assert "茅" * 400 not in json.dumps(
        provider.requests[1]["messages"][2].tool_results
    )


def test_tiny_token_budget_truncates_without_room_for_the_marker():
    async def run():
        tool = RecordingTool("get_kline", content="abcdefghij" * 20)
        registry = StubRegistry({"get_kline": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("call_1", "get_kline", {})), text_round("ok")]
        )
        loop = make_loop(
            provider, registry=registry, settings=make_settings(max_result_tokens=4)
        )
        await loop.run("短上限")
        return loop.messages, loop

    messages, loop = asyncio.run(run())

    content = json.loads(messages[2].tool_results[0][1])["content"]
    # 在 4-token 预算下没有空间容纳标记，因此改为裁剪前缀
    # 以适配预算，而不是超出预算。
    assert loop.TRUNCATION_MARKER not in content
    assert loop.memory.counter.count(content).tokens <= 4


def test_short_tool_result_is_backfilled_unchanged():
    async def run():
        tool = RecordingTool("get_quote", content="报价")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("call_1", "get_quote", {})), text_round("ok")]
        )
        loop = make_loop(
            provider, registry=registry, settings=make_settings(max_result_tokens=1000)
        )
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
        loop = make_loop(
            provider,
            registry=registry,
            settings=make_settings(max_turns=2),
            output=sink,
        )
        outcome = await loop.run("一直调用工具")
        return outcome, sink.events, loop.messages, provider

    outcome, events, messages, provider = asyncio.run(run())

    assert outcome.succeeded is False
    assert outcome.reason == "max_turns_exhausted"
    # 预算耗尽并非空跑：部分答案会说明
    # 原因，以及没有任何结论成立。
    assert "提前结束" in outcome.answer
    assert "轮次上限" in outcome.answer
    assert outcome.tool_calls == 2
    assert outcome.usage.input_tokens == 2
    assert kinds(events).count("error") == 1
    assert kinds(events).count("done") == 1
    # 部分答案会作为 ``answer`` 事件在 ``done`` 之前呈现。
    assert kinds(events)[-3:] == ["error", "answer", "done"]
    assert events[-3].data["reason"] == "max_turns_exhausted"
    assert events[-2].data["text"] == outcome.answer
    assert events[-1].data["succeeded"] is False
    assert events[-1].data["reason"] == "max_turns_exhausted"
    assert events[-1].data["tool_calls"] == 2
    assert len(provider.requests) == 2
    # 部分答案会写入该轮次的 assistant 消息，因此
    # 重新加载的会话仍能显示本次运行得出的结论。
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
        "tool_result",
        "assistant",
    ]
    assert messages[-1].content == outcome.answer


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
    assert events[1].data["usage"]["input_tokens"] == 0
    assert events[1].data["usage"]["output_tokens"] == 0


def test_provider_error_after_a_tool_round_keeps_the_tool_call_count():
    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("call_1", "get_quote", {})),
                RuntimeError("provider down"),
            ]
        )
        loop = make_loop(provider, registry=registry, output=sink)
        outcome = await loop.run("先查报价")
        return outcome, sink.events, loop.messages

    outcome, events, messages = asyncio.run(run())

    assert outcome.reason == "provider_error"
    assert outcome.tool_calls == 1
    assert kinds(events) == ["tool_status", "tool_status", "error", "done"]
    assert events[-1].data["tool_calls"] == 1
    assert events[-1].data["usage"]["input_tokens"] == 1
    assert events[-1].data["usage"]["output_tokens"] == 1
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool_result",
    ]


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
                tool_round(
                    ToolUse("call_1", "get_quote", {}),
                    ToolUse("call_2", "get_quote", {}),
                ),
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
    requested = {
        tool_use.call_id for message in second_request for tool_use in message.tool_uses
    }
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
            retry_policy=RetryPolicy(max_retries=4, base_delay_s=0.0, cap_delay_s=0.0),
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
            [
                RateLimitError("limited"),
                RateLimitError("limited"),
                RateLimitError("limited"),
                RateLimitError("limited"),
                RateLimitError("limited"),
            ]
        )
        loop = make_loop(
            provider,
            output=sink,
            retry_policy=RetryPolicy(max_retries=4, base_delay_s=0.0, cap_delay_s=0.0),
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
            retry_policy=RetryPolicy(max_retries=4, base_delay_s=0.0, cap_delay_s=0.0),
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
        loop = make_loop(provider, registry=registry, output=sink, stats=stats)
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
    # 写权限工具会被默认的 ReadOnlyGate 拒绝；重点在于
    # 该请求仍会被计数，且没有执行时间。
        registry = StubRegistry(
            {"get_quote": RecordingTool("get_quote", permission=PermissionLevel.WRITE)}
        )
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
    failed = [
        event.data
        for event in events
        if event.kind == "tool_status" and event.data["status"] == "failed"
    ]
    assert all("duration_ms" not in data for data in failed)


def test_failed_tool_result_reports_failed_status_with_duration():
    async def run():
        sink = Sink()
        tool = RecordingTool(
            "get_quote", content="stale", ok=False, error="denied by source"
        )
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

    completed = [
        event.data
        for event in events
        if event.kind == "tool_status" and event.data["status"] != "started"
    ]
    assert completed[0]["status"] == "failed"
    assert completed[0]["ok"] is False
    assert completed[0]["duration_ms"] == 250
    assert loop.stats.snapshot().tool_duration_ms == 250
    assert outcome.tool_duration_ms == 250


# -- 子 agent coordinator 注入 (docs 03.10) ------------------------------------

class NeedsCoordinatorTool:
    """工具替身，声明自己需要 coordinator，类似 write_report。"""

    name = "review_report"
    permission = PermissionLevel.READ
    timeout = None
    needs_coordinator = True
    coordinator = None

    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> ToolResult:
        self.calls.append({"coordinator": self.coordinator})
        return ToolResult(content="reviewed")


def test_loop_injects_the_coordinator_into_a_tool_that_needs_one():
    async def run():
        sink = Sink()
        tool = NeedsCoordinatorTool()
        registry = StubRegistry({"review_report": tool})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "review_report", {})),
                text_round("ok"),
            ]
        )
        sentinel = object()
        loop = make_loop(provider, registry=registry, output=sink, coordinator=sentinel)
        await loop.run("写报告")
        return tool, sentinel

    tool, sentinel = asyncio.run(run())

    assert tool.calls == [{"coordinator": sentinel}]


def test_loop_without_a_coordinator_leaves_the_tool_unwired():
    async def run():
        tool = NeedsCoordinatorTool()
        registry = StubRegistry({"review_report": tool})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "review_report", {})),
                text_round("ok"),
            ]
        )
        loop = make_loop(provider, registry=registry)
        await loop.run("写报告")
        return tool

    tool = asyncio.run(run())

    assert tool.calls == [{"coordinator": None}]


def test_done_event_reports_per_agent_usage():
    async def run():
        sink = Sink()
        stats = SessionStats()
        stats.record_agent_usage("risk", 42, 7)
        provider = ScriptedProvider([text_round("ok")])
        loop = make_loop(provider, output=sink, stats=stats)
        await loop.run("question")
        return sink.events

    events = asyncio.run(run())

    done = [event.data for event in events if event.kind == "done"][0]
    assert done["per_agent"]["risk"]["input_tokens"] == 42
    assert done["per_agent"]["risk"]["output_tokens"] == 7
    assert done["per_agent"]["risk"]["runs"] == 1


class RecordingSpawnTool:
    """spawn_agent 的工具替身：记录交给它的每一批任务。"""

    name = "spawn_agent"
    permission = PermissionLevel.READ
    timeout = None
    needs_coordinator = True
    coordinator = None

    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(content="ok")


def test_loop_guard_does_not_block_fan_out_with_different_tasks():
    """spawn_agent 每批以不同的参数被调用一次 (docs 03.10)。

    重复调用防护以 (tool, args) 为键，因此不同的任务列表是不同的调用。
    这里把它固定下来：若防护仅以工具名为键，第二批就会被
    当成循环而被拒绝。
    """
    async def run():
        tool = RecordingSpawnTool()
        registry = StubRegistry({"spawn_agent": tool})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "spawn_agent", {"tasks": ["甲", "乙"]})),
                tool_round(ToolUse("c2", "spawn_agent", {"tasks": ["丙", "丁"]})),
                tool_round(ToolUse("c3", "spawn_agent", {"tasks": ["戊"]})),
                text_round("全部完成"),
            ]
        )
        loop = make_loop(provider, registry=registry)
        outcome = await loop.run("分批摘要")
        return outcome, tool.calls

    outcome, calls = asyncio.run(run())

    # 三批任务全部运行；没有任何一批被防护拒绝。
    assert len(calls) == 3
    assert calls[0]["tasks"] == ["甲", "乙"]
    assert calls[2]["tasks"] == ["戊"]
    assert outcome.succeeded is True


def test_done_event_per_agent_defaults_to_empty():
    async def run():
        sink = Sink()
        loop = make_loop(ScriptedProvider([text_round("ok")]), output=sink)
        await loop.run("question")
        return sink.events

    events = asyncio.run(run())

    done = [event.data for event in events if event.kind == "done"][0]
    assert done["per_agent"] == {}


class LazyRegistry:
    """最小 registry 替身，建模 resident/lazy 的划分。

    loop 的 lazy 门控在存在时读取 ``lazy_names`` 与 ``is_active``，
    因此无需构建完整的工具目录即可验证真实 registry 的两阶段规则。
    """

    def __init__(self, tools, *, lazy: set[str] | None = None):
        self.tools = dict(tools)
        self._lazy = set(lazy or ())
        self._active = {name for name in self.tools if name not in self._lazy}

    def names(self):
        return list(self.tools)

    def resolve(self, name):
        return self.tools.get(name)

    def schemas(self, names: set[str] | None = None):
        selected = self._active if names is None else names
        return [
            {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
            for n in self.tools
            if n in selected
        ]

    def is_read_only(self, name):
        return True

    def lazy_names(self):
        return [n for n in self.tools if n in self._lazy]

    def is_active(self, name):
        return name in self._active

    def activate(self, name):
        if name not in self.tools or name in self._active:
            return False
        self._active.add(name)
        return True


def test_a_direct_call_to_a_lazy_tool_activates_it_and_runs():
    """按需注入是省 token 的手段，不是权限。

    模型一旦点名调用某个按需工具，就说明它已经知道要什么；此时拒绝只会白费一轮。
    循环就地激活并放行，同时留下 ``tool_activated`` 事件使这次激活可归因。
    """
    async def run():
        sink = Sink()
        tool = RecordingTool("calc_valuation", content="估值")
        registry = LazyRegistry({"calc_valuation": tool}, lazy={"calc_valuation"})
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "calc_valuation", {"symbol": "600519"})),
                text_round("完成"),
            ]
        )
        loop = make_loop(provider, registry=registry, output=sink)
        await loop.run("估值")
        return tool, registry, sink.events

    tool, registry, events = asyncio.run(run())

    # 本次就执行了，且注册表记住了这次激活。
    assert len(tool.calls) == 1
    assert registry.is_active("calc_valuation") is True
    activated = [e.data.get("name") for e in events if e.kind == "tool_activated"]
    assert activated == ["calc_valuation"]


def test_activated_lazy_tool_runs_normally():
    async def run():
        sink = Sink()
        tool = RecordingTool("calc_valuation", content="估值")
        registry = LazyRegistry({"calc_valuation": tool}, lazy={"calc_valuation"})
        registry.activate("calc_valuation")
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "calc_valuation", {"symbol": "600519"})),
                text_round("完成"),
            ]
        )
        loop = make_loop(provider, registry=registry, output=sink)
        await loop.run("估值")
        return tool

    tool = asyncio.run(run())

    assert len(tool.calls) == 1


class ProgressTool:
    """声明 ``needs_progress`` 的工具替身：执行期间上报一次进展。"""

    permission = PermissionLevel.READ
    timeout = None
    needs_progress = True
    progress = None

    def __init__(self, name: str = "slow_tool"):
        self.name = name
        self.reported = False

    async def run(self, **kwargs) -> ToolResult:
        # 工具只描述进展内容，调用标识由循环补齐——否则每个长任务工具都得
        # 自己维护 call_id，并把对话层概念泄漏进工具实现。
        if self.progress is not None:
            await self.progress({"phase": "panel", "fetched": 10, "total": 20})
            self.reported = True
        return ToolResult(content="done", ok=True)


def test_progress_tool_reports_through_the_loop():
    """长任务工具的上报必须变成引擎事件。

    服务端心跳是 SSE 注释帧，客户端的空闲看门狗只在真实事件到达时重置；因此
    静默数分钟的工具若不产生事件，界面会在"运行中"被看门狗掐断（docs 03.12）。
    """
    async def run():
        sink = Sink()
        tool = ProgressTool()
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "slow_tool", {})),
                text_round("完成"),
            ]
        )
        loop = make_loop(provider, registry=StubRegistry({"slow_tool": tool}), output=sink)
        await loop.run("长任务")
        return tool, sink.events

    tool, events = asyncio.run(run())

    assert tool.reported is True
    progress = [e for e in events if e.kind == "tool_progress"]
    assert len(progress) == 1
    # 循环补齐了调用标识，工具自身的载荷原样保留。
    assert progress[0].data["call_id"] == "c1"
    assert progress[0].data["name"] == "slow_tool"
    assert progress[0].data["fetched"] == 10


def test_progress_failure_does_not_break_the_tool():
    """进展通道断掉不得连累正在进行的计算：上报失败只记日志，工具照常完成。"""
    class FlakySink(Sink):
        async def emit(self, event: EngineEvent) -> None:
            if event.kind == "tool_progress":
                raise RuntimeError("progress transport gone")
            await super().emit(event)

    async def run():
        tool = ProgressTool()
        sink = FlakySink()
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "slow_tool", {})),
                text_round("完成"),
            ]
        )
        loop = make_loop(provider, registry=StubRegistry({"slow_tool": tool}), output=sink)
        outcome = await loop.run("长任务")
        return outcome, sink.events

    outcome, events = asyncio.run(run())

    assert outcome.answer == "完成"
    # 关键判别点：工具以成功收尾。若上报异常逸出，循环会把它当作工具失败
    # （completed/failed + ok=false），即使最终答案仍然能凑出来。
    completed = [
        e for e in events
        if e.kind == "tool_status" and e.data.get("name") == "slow_tool"
        and e.data.get("status") != "started"
    ]
    assert completed and completed[0].data["ok"] is True
