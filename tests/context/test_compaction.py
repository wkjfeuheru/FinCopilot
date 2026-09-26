"""自动 compaction：触发、digest、降级与审计轨迹。"""

import asyncio
import json

from finharness.config.settings import ContextSettings, Settings
from finharness.context.compaction import AutoCompactor
from finharness.context.memory.working import WorkingMemory
from finharness.context.session import ResearchContext
from finharness.context.tokens import TokenCounter
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import HookChain
from finharness.permissions.gate import PermissionGate
from finharness.types import Msg, ToolUse
from tests.conftest import settings_with_cache

# 共享词表缓存：获取代价高昂，不能在每个测试中重复进行。
COUNTER = TokenCounter()

# 复用 engine 的测试替身；本测试套件没有 tests 包。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import (  # noqa: E402
    ScriptedProvider,
    Sink,
    StubRegistry,
    text_round,
)


def make_settings(tmp_path, **context) -> Settings:
    values = {"context_window_tokens": 600, "compaction_ratio": 0.5}
    values.update(context)
    return settings_with_cache(tmp_path, context=ContextSettings(**values))


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


def test_compaction_is_an_explicit_phase_before_thinking(tmp_path):
    async def run_forced_compaction():
        provider = ScriptedProvider([text_round("摘要"), text_round("答案")])
        sink = Sink()
        loop = build_loop(tmp_path, provider, output=sink)
        for index in range(20):
            loop.memory.append_user("撑满窗口的历史" * 5)
            loop.memory.append_assistant(Msg(role="assistant", content="阶段回答" * 5))
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run_forced_compaction())
    phases = [e.data["phase"] for e in events if e.kind == "state"]
    assert phases[:3] == ["hydrate", "compact", "thinking"]


def test_compaction_fires_when_the_window_is_exceeded(tmp_path):
    """较长的 transcript 应在请求发出前被折叠。"""
    provider = ScriptedProvider([text_round("ok")])

    async def run():
        loop = build_loop(tmp_path, provider)
        # 预加载历史，使其超出（较小的）窗口。
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

    assert list(loop.compactions) == []


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

    # 不产生 digest 消息：更早的历史被移到 summary layer，
    # 通过 system prompt 注入，而不是伪装成 user input。
    contents = [m.content or "" for m in loop.memory.raw]
    assert not any("摘要内容" in text for text in contents)
    assert not any("历史问题0" in text for text in contents), "oldest history left the window"
    assert loop.summary is not None
    assert loop.summary.segments, "folded history becomes a summary segment"


def test_summary_segment_carries_a_data_ledger(tmp_path):
    """Compaction 会移除 tool result，因此 ledger 是模型仍然
    知道某次取数已经发生过的依据。"""
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
    """摘要器故障不得中断本轮对话（文档 3.6.3）。"""
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    memory = WorkingMemory(ctx=ctx, settings=settings, counter=COUNTER)
    for index in range(20):
        memory.append_user("撑满窗口的历史内容" * 4)
        # 一个 round 即一次交互，因此每轮都需要对应的 assistant frame。
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
    """摘要调用总是失败的 Provider。"""

    async def stream(self, **kwargs):
        raise RuntimeError("摘要模型不可用")
        yield  # pragma: no cover


def test_unreachable_threshold_is_reported_not_silent(tmp_path):
    """阈值低于不可压缩地板时必须如实告警，而不是每轮空转。

    窗口的固定开销（system + 工具 schema）压缩删不掉。这里特意把 system 提示
    写得比阈值还大，于是「折叠到最近一轮」之后窗口仍在地板之上；正确的行为是
    出一条说明配置过低的 warning，而不是假装压缩成功。
    """
    settings = make_settings(tmp_path, context_window_tokens=100, compaction_ratio=0.5)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    memory = WorkingMemory(ctx=ctx, settings=settings, counter=COUNTER)
    for index in range(6):
        memory.append_user(f"历史问题{index}" * 4)
        memory.append_assistant(Msg(role="assistant", content=f"历史回答{index}" * 4))

    compactor = AutoCompactor(
        provider=ScriptedProvider([text_round("摘要内容")]),
        memory=memory,
        settings=settings,
        system="一段足够长的系统提示词，其长度本身已经超过压缩阈值" * 3,
    )
    result = asyncio.run(compactor.compact())

    assert result.after_tokens >= compactor.memory.compaction_threshold()
    assert result.warning and "窗口配置过低" in result.warning


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
    # 累计计费数值是独立的，不受 compaction 影响。
    assert "usage" in done


def test_compaction_summary_is_observed_and_accounted(tmp_path):
    """压缩摘要本身是一次 LLM 调用：必须带上 call_type=compaction，并回填 token。

    此前这次调用的 token 完全不计入会话成本，算是成本视图里的一个真实缺口。
    """
    from finharness.observability.metrics import MetricsRecorder
    from finharness.observability.observer import Observer

    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    memory = WorkingMemory(ctx=ctx, settings=settings, counter=COUNTER)
    memory.append_user("问题")
    for index in range(10):
        memory.append_assistant(
            Msg(role="assistant", content=None, tool_uses=[ToolUse(f"c{index}", "get_quote", {"symbol": "600519"})])
        )
        memory.append(Msg(role="tool_result", content=None, tool_results=[(f"c{index}", "结果" * 20)]))

    metrics = MetricsRecorder()
    observed_usages: list[tuple[int, int]] = []
    compactor = AutoCompactor(
        provider=ScriptedProvider([text_round("摘要内容", input_tokens=40, output_tokens=8)]),
        memory=memory,
        settings=settings,
        observer=Observer(metrics=metrics),
        on_usage=lambda i, o: observed_usages.append((i, o)),
    )
    asyncio.run(compactor.compact())

    assert observed_usages == [(40, 8)]
    rendered = metrics.render().decode()
    assert 'llm_tokens_total{call_type="compaction",kind="input"' in rendered
    assert "40.0" in rendered


# -- 转录稿裁剪：按 token 而非字符 ---------------------------------------------


def test_the_transcript_clip_is_token_based_and_settings_driven():
    """转录稿过去把每条结果硬编码截到 800 字符。

    在按 token 计量的体系里，那是一个更紧、又与语言无关的第二个上限——中文下
    800 字符仅约 470 token，一条刚在上下文预算下幸存的结果会在压缩时被砍得更短。
    现在它按 token 预算裁剪，因此可以断言真实 token 数。
    """
    from finharness.context.compaction import _render_transcript

    settings = Settings(context=ContextSettings(context_window_tokens=600, compaction_ratio=0.5))
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    memory = WorkingMemory(ctx=ctx, settings=settings, counter=COUNTER)
    for index in range(5):
        memory.append(Msg(role="user", content=f"问题{index}"))
        memory.append(
            Msg(role="assistant", content=None, tool_uses=[ToolUse(f"c{index}", "get_quote", {})])
        )
        memory.append(
            Msg(role="tool_result", content=None, tool_results=[(f"c{index}", "内容" * 2000)])
        )

    transcript = _render_transcript(
        memory.raw, counter=COUNTER, max_result_tokens=200
    )

    # 每条结果都被裁到预算内：5 条各约 200 token，总计远低于未裁剪的量。
    assert COUNTER.count(transcript).tokens < 200 * 5 + 200
    # 裁剪是显式的，而不是静默丢弃。
    assert "已截断" in transcript


def test_a_zero_budget_keeps_the_transcript_verbatim():
    """预算为 0 表示"不裁剪"，而不是"裁到空"。"""
    from finharness.context.compaction import _render_transcript

    settings = Settings(context=ContextSettings(context_window_tokens=600, compaction_ratio=0.5))
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    memory = WorkingMemory(ctx=ctx, settings=settings, counter=COUNTER)
    memory.append(Msg(role="user", content="问题"))
    memory.append(
        Msg(role="assistant", content=None, tool_uses=[ToolUse("c1", "get_quote", {})])
    )
    memory.append(Msg(role="tool_result", content=None, tool_results=[("c1", "原始结果")]))

    transcript = _render_transcript(memory.raw, counter=COUNTER, max_result_tokens=0)

    assert "原始结果" in transcript
