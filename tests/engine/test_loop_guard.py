"""Loop guard: repeated identical tool calls are nudged, then aborted.

A repeated identical call cannot add information — its result is already in the
transcript and in the cache — so the guard refuses it and, if the model repeats
the same mistake, ends the run with whatever was already established.
"""

import asyncio
import json
import sys
from pathlib import Path

from finharness.config.settings import ContextSettings, Settings
from finharness.context.session import ResearchContext, PlanStep
from finharness.context.tokens import TokenCounter
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import HookChain
from finharness.permissions.gate import PermissionGate
from finharness.types import ToolUse

# Shared vocabulary cache: fetching it is expensive and must not repeat per test.
COUNTER = TokenCounter()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import RecordingTool, ScriptedProvider, Sink, StubRegistry, text_round, tool_round  # noqa: E402


def make_settings(tmp_path, **context) -> Settings:
    values = {"max_identical_tool_calls": 3, "max_turns": 10}
    values.update(context)
    return Settings(context=ContextSettings(**values), data={"cache_dir": tmp_path / "cache"})


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
    """The same tool call, issued `times` times, then a final answer."""
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

    # The third identical call is refused, so the tool ran only twice.
    assert len(tool.calls) == 2
    assert outcome.succeeded is True  # the model recovered and answered


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
    """The refused call still gets a paired tool_result so the transcript stays valid."""
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(3))
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}))

    asyncio.run(loop.run("查报价"))
    results = [m for m in loop.messages if m.role == "tool_result"]

    # Every assistant tool call has a matching result id.
    requested = {t.call_id for m in loop.messages for t in m.tool_uses}
    answered = {cid for m in results for cid, _ in m.tool_results}
    assert requested == answered == {"call_0", "call_1", "call_2"}


def test_distinct_arguments_are_not_treated_as_repeats(tmp_path):
    """Comparing two symbols with the same tool is legitimate, not a loop."""
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
    """A/B/A/B is a loop too, so counting must be cumulative, not consecutive."""
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
    # Each shape was refused on its third attempt.
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
    # The run ends with what it established rather than an empty answer.
    assert outcome.answer, "a detected loop must still report partial findings"
    assert "提前结束" in outcome.answer
    kinds = [e.kind for e in sink.events]
    assert "loop_detected" not in kinds  # surfaced as error/done, not a new kind
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
    """With a plan in place the nudge should point at research_plan, not just reuse."""
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(repeated_rounds(3))
    loop = build_loop(tmp_path, provider, registry=StubRegistry({"get_quote": tool}), plan=True)

    asyncio.run(loop.run("复杂任务"))
    refused = loop.messages[-1] if loop.messages else None

    # Find the refusal payload among the tool results.
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
    """The guard is scoped to one request, so a later question starts clean."""
    tool = RecordingTool("get_quote", content="报价")

    async def run():
        # Each run issues the same identical call twice; with a threshold of 3
        # neither run reaches the nudge, proving counts restarted.
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
    # Both identical calls executed: if counts carried over, the second would
    # have been refused.
    assert len(tool.calls) == 2


def test_turn_counter_resets_per_run(tmp_path):
    tool = RecordingTool("get_quote", content="报价")

    async def run():
        # Each run is one tool round plus a final answer = 2 turns.
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

    # Both runs take 2 turns; without a per-run reset the second would read 4.
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

    # Threshold 2: the second identical call is the first offence.
    assert len(tool.calls) == 1
    assert outcome.succeeded is False
