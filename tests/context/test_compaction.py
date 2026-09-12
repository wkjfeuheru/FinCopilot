"""Auto-compaction: trigger, digest, degradation, and the audit trail."""

import asyncio
import json

from finharness.config.settings import ContextSettings, Settings
from finharness.context.compaction import AutoCompactor
from finharness.context.memory.working import WorkingMemory
from finharness.context.session import ResearchContext
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import HookChain
from finharness.permissions.gate import PermissionGate
from finharness.context.tokens import TokenCounter
from finharness.types import Msg, ModelUsage, StreamChunk, StreamEvent, ToolUse

# Shared vocabulary cache: fetching it is expensive and must not repeat per test.
COUNTER = TokenCounter()

# Reuse the engine doubles; the suite has no tests package.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import (  # noqa: E402
    ScriptedProvider,
    Sink,
    StubRegistry,
    RecordingTool,
    make_loop,
    text_round,
    tool_round,
)


def make_settings(tmp_path, **context) -> Settings:
    values = {"context_window_tokens": 600, "compaction_ratio": 0.5}
    values.update(context)
    return Settings(context=ContextSettings(**values), data={"cache_dir": tmp_path / "cache"})


def build_loop(tmp_path, provider, *, registry=None, output=None, settings=None, hooks=None):
    settings = settings or make_settings(tmp_path)
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
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


def test_compaction_fires_when_the_window_is_exceeded(tmp_path):
    """A long transcript should be folded before the request goes out."""
    provider = ScriptedProvider([text_round("ok")])

    async def run():
        loop = build_loop(tmp_path, provider)
        # Preload history past the (small) window.
        for index in range(20):
            loop.memory.append_user("一段足以把窗口撑满的历史内容" * 4)
            loop.memory.append_assistant(Msg(role="assistant", content="阶段回答" * 4))
        outcome = await loop.run("新问题")
        return loop, outcome

    loop, outcome = asyncio.run(run())

    assert outcome.succeeded is True
    assert loop.compactions, "expected at least one compaction"
    result = loop.compactions[0]
    assert result.compacted is True
    assert result.degraded is False
    assert result.after_tokens < result.before_tokens


def test_compaction_is_skipped_below_the_threshold(tmp_path):
    provider = ScriptedProvider([text_round("ok")])

    async def run():
        loop = build_loop(tmp_path, provider, settings=make_settings(tmp_path, context_window_tokens=100000))
        loop.memory.append_user("短历史")
        await loop.run("新问题")
        return loop

    loop = asyncio.run(run())

    assert loop.compactions == []


def test_summary_replaces_the_middle_and_keeps_recent_rounds(tmp_path):
    provider = ScriptedProvider([text_round("摘要内容")])

    async def run():
        loop = build_loop(tmp_path, provider)
        for index in range(15):
            loop.memory.append_user(f"历史问题{index}" * 8)
            loop.memory.append_assistant(Msg(role="assistant", content=f"历史回答{index}" * 8))
        await loop.run("当前问题")
        return loop

    loop = asyncio.run(run())

    # No digest message: earlier history moved to the summary layer, which is
    # injected through the system prompt instead of masquerading as user input.
    contents = [m.content or "" for m in loop.memory.raw]
    assert not any("摘要内容" in text for text in contents)
    assert not any("历史问题0" in text for text in contents), "oldest history left the window"
    assert loop.summary is not None
    assert loop.summary.segments, "folded history becomes a summary segment"


def test_summary_segment_carries_a_data_ledger(tmp_path):
    """Compaction removes the tool results, so the ledger is how the model still
    knows a fetch already happened."""
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    memory = WorkingMemory(ctx=ctx, settings=settings, counter=COUNTER)
    memory.append_user("问题")
    for index in range(10):
        memory.append_assistant(
            Msg(role="assistant", content=None, tool_uses=[ToolUse(f"c{index}", "get_quote", {"symbol": "600519"})])
        )
        memory.append(Msg(role="tool_result", content=None, tool_results=[(f"c{index}", "结果" * 20)]))

    compactor = AutoCompactor(
        provider=ScriptedProvider([text_round("摘要内容")]), memory=memory, settings=settings
    )
    result = asyncio.run(compactor.compact())

    assert result.compacted is True
    assert "get_quote(600519)" in result.ledger
    assert result.seq_from >= 1 and result.seq_to >= result.seq_from


def test_summarizer_failure_degrades_without_blocking(tmp_path):
    """A broken summariser must not stop the turn (docs 3.6.3)."""
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    memory = WorkingMemory(ctx=ctx, settings=settings, counter=COUNTER)
    for index in range(20):
        memory.append_user("撑满窗口的历史内容" * 4)
        # A round is an exchange, so each needs its assistant frame.
        memory.append_assistant(Msg(role="assistant", content="阶段回答" * 4))
    compactor = AutoCompactor(
        provider=FailingProvider(), memory=memory, settings=settings
    )

    result = asyncio.run(compactor.compact())

    assert result.degraded is True
    assert result.warning and "降级" in result.warning
    assert result.compacted is True, "the fallback must still reduce the window"
    assert result.after_tokens < result.before_tokens


class FailingProvider:
    """Provider whose summarisation call always fails."""

    async def stream(self, **kwargs):
        raise RuntimeError("摘要模型不可用")
        yield  # pragma: no cover


def test_compaction_event_is_emitted(tmp_path):
    provider = ScriptedProvider([text_round("摘要"), text_round("答案")])
    sink = Sink()

    async def run():
        loop = build_loop(tmp_path, provider, output=sink)
        for index in range(20):
            loop.memory.append_user("撑满窗口的历史" * 5)
            loop.memory.append_assistant(Msg(role="assistant", content="阶段回答" * 5))
        await loop.run("问题")
        return sink

    sink = asyncio.run(run())

    kinds = [event.kind for event in sink.events]
    assert "context_compacted" in kinds
    payload = next(e.data for e in sink.events if e.kind == "context_compacted")
    assert payload["before_tokens"] >= payload["after_tokens"]


def test_compaction_is_audited(tmp_path):
    writer = AuditLogWriter(tmp_path / "audit.jsonl")
    audit = AuditHook(writer, session_id="s")
    provider = ScriptedProvider([text_round("摘要"), text_round("答案")])

    async def run():
        loop = build_loop(tmp_path, provider, hooks=HookChain([audit]))
        for index in range(20):
            loop.memory.append_user("撑满窗口的历史" * 5)
            loop.memory.append_assistant(Msg(role="assistant", content="阶段回答" * 5))
        await loop.run("问题")

    asyncio.run(run())
    records = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]

    assert any(record["action"] == "compact" for record in records)


def test_done_payload_reports_window_and_compaction_count(tmp_path):
    provider = ScriptedProvider([text_round("摘要"), text_round("答案")])
    sink = Sink()

    async def run():
        loop = build_loop(tmp_path, provider, output=sink)
        for index in range(20):
            loop.memory.append_user("撑满窗口的历史" * 5)
            loop.memory.append_assistant(Msg(role="assistant", content="阶段回答" * 5))
        await loop.run("问题")
        return sink

    sink = asyncio.run(run())

    done = next(e.data for e in sink.events if e.kind == "done")
    assert "window_tokens" in done
    assert done["compactions"] >= 1
    # The cumulative billing figure is separate and unaffected by compaction.
    assert "usage" in done
