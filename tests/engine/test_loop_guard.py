"""Loop guard：重复的相同 tool call 会先被提醒，随后被中止。

重复的相同调用无法带来新信息——其结果已经在 transcript 和缓存中——因此
guard 会拒绝它，并且如果模型重犯同样的错误，就以已经确立的内容结束本次运行。
"""

import asyncio
import json
import sys
from pathlib import Path

from finharness.config.settings import ContextSettings, Settings
from finharness.context.session import PlanStep, ResearchContext
from finharness.context.tokens import TokenCounter
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import HookChain
from finharness.permissions.gate import PermissionGate
from finharness.types import ToolUse
from tests.conftest import settings_with_cache

# 共享的词汇表缓存：抓取它代价高昂，绝不能在每个测试中重复进行。
COUNTER = TokenCounter()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import (  # noqa: E402
    RecordingTool,
    ScriptedProvider,
    Sink,
    StubRegistry,
    text_round,
    tool_round,
)


def make_settings(tmp_path, **context) -> Settings:
    values = {"max_identical_tool_calls": 3, "max_turns": 10}
    values.update(context)
    return settings_with_cache(tmp_path, context=ContextSettings(**values))


def build_loop(tmp_path, provider, *, registry=None, output=None, settings=None, hooks=None, plan=False):
    settings = settings or make_settings(tmp_path)
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    if plan:
        ctx.set_plan("研究任务", [PlanStep(seq=1, action="取数")])
    return AgentLoop(
        provider=provider,
        registry=registry or StubRegistry(),
        settings=settings,
        system="系统提示",
        output=output,
        cite=cite,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        hooks=hooks,
        counter=COUNTER,
    )


def repeated_rounds(times: int, args: dict | None = None) -> list[list]:
    """同一个 tool call，重复发起 `times` 次，然后给出最终回答。"""
    call_args = args or {"symbol": "600519"}
    rounds = [
        tool_round(ToolUse(f"call_{index}", "get_quote", dict(call_args)))
        for index in range(times)
    ]
    rounds.append(text_round("最终回答"))
    return rounds


def test_first_repeat_is_refused_with_a_nudge_not_executed(tmp_path):
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(3))
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}))

    outcome = asyncio.run(loop.run("查报价"))

    # 第三次相同调用被拒绝，因此工具只运行了两次。
    assert len(tool.calls) == 2
    assert outcome.succeeded is True  # 模型恢复正常并作出了回答


def test_nudge_message_tells_the_model_to_reuse(tmp_path):
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(3))
    sink = Sink()
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}), output=sink)

    asyncio.run(loop.run("查报价"))
    events = [e for e in sink.events if e.kind == "loop_guard"]

    assert events, "the guard must be visible"
    assert events[0].data["action"] == "refused"


def test_guard_events_and_results_are_well_formed(tmp_path):
    """被拒绝的调用仍会获得配对的 tool_result，以保持 transcript 合法。"""
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(3))
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}))

    asyncio.run(loop.run("查报价"))
    results = [m for m in loop.messages if m.role == "tool_result"]

    # 每个 assistant tool call 都有匹配的 result id。
    requested = {t.call_id for m in loop.messages for t in m.tool_uses}
    answered = {cid for m in results for cid, _ in m.tool_results}
    assert requested == answered == {"call_0", "call_1", "call_2"}


def test_distinct_arguments_are_not_treated_as_repeats(tmp_path):
    """用同一个工具比较两个标的属于正常行为，并非循环。"""
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(
        [
            tool_round(ToolUse("c1", "get_quote", {"symbol": "600519"})),
            tool_round(ToolUse("c2", "get_quote", {"symbol": "000858"})),
            tool_round(ToolUse("c3", "get_quote", {"symbol": "600519"})),
            tool_round(ToolUse("c4", "get_quote", {"symbol": "000858"})),
            text_round("对比完成"),
        ]
    )
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}))

    outcome = asyncio.run(loop.run("对比两只标的"))

    assert outcome.succeeded is True
    assert len(tool.calls) == 4, "different symbols must all be allowed through"


def test_alternating_loop_is_caught_by_cumulative_counting(tmp_path):
    """A/B/A/B 同样是循环，因此计数必须是累计的，而非连续计数。"""
    tool = RecordingTool("get_quote", content="报价")
    other = RecordingTool("get_kline", content="K线")
    provider = ScriptedProvider(
        [
            tool_round(ToolUse("a1", "get_quote", {"symbol": "600519"})),
            tool_round(ToolUse("b1", "get_kline", {"symbol": "600519"})),
            tool_round(ToolUse("a2", "get_quote", {"symbol": "600519"})),
            tool_round(ToolUse("b2", "get_kline", {"symbol": "600519"})),
            tool_round(ToolUse("a3", "get_quote", {"symbol": "600519"})),
            text_round("答案"),
        ]
    )
    loop = build_loop(
        tmp_path, provider, registry=StubRegistry({"get_quote": tool, "get_kline": other})
    )

    outcome = asyncio.run(loop.run("交替调用"))

    assert outcome.succeeded is True
    # 每种调用形态都在其第三次尝试时被拒绝。
    assert len(tool.calls) == 2
    assert len(other.calls) == 2


def test_second_offence_aborts_the_run_with_partial_findings(tmp_path):
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(5))
    sink = Sink()
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}), output=sink)

    outcome = asyncio.run(loop.run("查报价"))

    assert outcome.succeeded is False
    assert outcome.reason == "loop_detected"
    # 运行以已确立的结论结束，而非给出空答案。
    assert outcome.answer, "a detected loop must still report partial findings"
    assert "提前结束" in outcome.answer
    kinds = [e.kind for e in sink.events]
    assert "loop_detected" not in kinds  # 以 error/done 呈现，而不是新的事件类型
    assert kinds.count("done") == 1
    assert any(e.kind == "error" and e.data.get("reason") == "loop_detected" for e in sink.events)


def test_partial_findings_include_conclusions_and_citation_count(tmp_path):
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(5))
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}))
    loop.ctx.add_conclusion("已确认报价为 100 元", ["cit_000001"])

    outcome = asyncio.run(loop.run("查报价"))

    assert "已确认报价为 100 元" in outcome.answer
    assert "cit_000001" in outcome.answer


def test_complex_task_nudge_mentions_replanning(tmp_path):
    """在已有 plan 的情况下，提醒应指向 research_plan，而不仅仅是复用。"""
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(3))
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}), plan=True)

    asyncio.run(loop.run("复杂任务"))
    refused = loop.messages[-1] if loop.messages else None

    # 在各 tool result 中找出拒绝 payload。
    payloads = [
        json.loads(raw)
        for m in loop.messages
        if m.role == "tool_result"
        for _, raw in m.tool_results
    ]
    refusals = [p for p in payloads if p["ok"] is False and "重复" in (p["error"] or "")]
    assert refusals, "expected a refusal payload"
    assert "research_plan" in refusals[0]["error"]
    assert refused is not None


def test_simple_task_nudge_does_not_mention_replanning(tmp_path):
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(3))
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}))

    asyncio.run(loop.run("查报价"))
    payloads = [
        json.loads(raw)
        for m in loop.messages
        if m.role == "tool_result"
        for _, raw in m.tool_results
    ]
    refusals = [p for p in payloads if p["ok"] is False and "重复" in (p["error"] or "")]

    assert refusals
    assert "research_plan" not in refusals[0]["error"]


def test_counts_reset_between_runs(tmp_path):
    """guard 的作用域限于单次请求，因此后续提问会从零开始。"""
    tool = RecordingTool("get_quote", content="报价")

    async def run():
        # 每次运行都将同一相同调用发起两次；在阈值为 3 的情况下，
        # 两次运行都达不到提醒阈值，从而证明计数已重置。
        provider = ScriptedProvider(
            [tool_round(ToolUse("a1", "get_quote", {"symbol": "600519"})), text_round("第一次答")]
            + [tool_round(ToolUse("a2", "get_quote", {"symbol": "600519"})), text_round("第二次答")]
        )
        loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}))
        first = await loop.run("第一次")
        second = await loop.run("第二次")
        return first, second, tool

    first, second, tool = asyncio.run(run())

    assert first.succeeded is True
    assert second.succeeded is True
    # 两次相同调用都执行了：如果计数被延续，那么第二次
    # 本应被拒绝。
    assert len(tool.calls) == 2


def test_turn_counter_resets_per_run(tmp_path):
    tool = RecordingTool("get_quote", content="报价")

    async def run():
        # 每次运行是一轮 tool round 加一个最终回答 = 2 turns。
        rounds = [
            tool_round(ToolUse("a1", "get_quote", {"symbol": "600519"})),
            text_round("第一答"),
            tool_round(ToolUse("b1", "get_quote", {"symbol": "000858"})),
            text_round("第二答"),
        ]
        provider = ScriptedProvider(rounds)
        loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}))
        await loop.run("第一次")
        first_turn = loop.turn
        await loop.run("第二次")
        return first_turn, loop.turn

    first_turn, second_turn = asyncio.run(run())

    # 两次运行都耗时 2 turns；若无每次运行的重置，第二次会读到 4。
    assert first_turn == 2
    assert second_turn == 2


def test_detection_is_audited(tmp_path):
    writer = AuditLogWriter(tmp_path / "audit.jsonl")
    audit = AuditHook(writer, session_id="s")
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(5))
    loop = build_loop(
        tmp_path, provider, registry=StubRegistry({"get_quote": tool}),
        hooks=HookChain([audit]),
    )

    asyncio.run(loop.run("查报价"))
    records = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]

    assert any(record["action"] == "loop_detected" for record in records)


def test_threshold_is_configurable(tmp_path):
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(4))
    loop = build_loop(
        tmp_path,
        provider,
        registry=StubRegistry({"get_quote": tool}),
        settings=make_settings(tmp_path, max_identical_tool_calls=2),
    )

    outcome = asyncio.run(loop.run("查报价"))

    # 阈值为 2：第二次相同调用即为第一次违规。
    assert len(tool.calls) == 1
    assert outcome.succeeded is False
