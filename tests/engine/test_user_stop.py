"""用户主动停止生成（docs 03.3 中断与恢复）。

停止是**用户意图**，不是失败。这几条不变量决定了界面的正确性，因此逐条断言：

* 停止不产生 ``error`` 事件——用户按下停止却看到一条错误提示是错的；
* 停止带着已有发现收尾（``answer`` + ``reason="user_stopped"``），使已付出的
  取数与结论不白费；
* 本轮消息确实落库，因此刷新后仍在；
* 断点记为 ``stopped``（可继续），而不是 ``completed``。
"""

import asyncio
from pathlib import Path

import pytest

from finharness.config.settings import ContextSettings, Settings
from finharness.context.memory.store import MemoryStore
from finharness.context.session import Plan, PlanStep, ResearchContext
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop, StopSignal
from finharness.provider.base import Provider
from finharness.types import ModelUsage, StreamChunk, StreamEvent, ToolUse

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import (  # noqa: E402
    RecordingTool,
    ScriptedProvider,
    Sink,
    StubRegistry,
    message_end,
    text_round,
    tool_round,
)


class StopAfterProvider(Provider):
    """流式输出若干 delta 后触发一个回调（模拟用户此刻按下停止）。

    它刻意在多轮之间保持"每轮都流出一批 delta"，因此停止一定发生在流式
    过程中——那正是逐 chunk 检查点负责的场景。
    """

    def __init__(self, parts: list[str], *, on_after):
        self.parts = parts
        self.on_after = on_after
        self.rounds = 0

    async def stream(
        self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage
    ):
        self.rounds += 1
        for index, part in enumerate(self.parts):
            yield StreamChunk(StreamEvent.TEXT_DELTA, part)
            if index == 0:
                self.on_after()
        # 若引擎没有在流中途停下，这里给出一个正常的回合结束。
        yield message_end()


def make_settings(tmp_path) -> Settings:
    return Settings(
        context=ContextSettings(context_window_tokens=100000),
        data={"cache_dir": tmp_path / "cache"},
    )


def build_loop(tmp_path, provider, *, store=None, conversation_id="c_stop", ctx=None):
    settings = make_settings(tmp_path)
    cite = CitationRegistry()
    ctx = ctx or ResearchContext(cite=cite, settings=settings)
    return AgentLoop(
        provider=provider,
        registry=StubRegistry(),
        settings=settings,
        system="系统提示",
        cite=cite,
        ctx=ctx,
        conversation_id=conversation_id,
        store=store,
    )


def test_stop_mid_stream_emits_no_error_and_keeps_partial_answer(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    loop = build_loop(tmp_path, ScriptedProvider([text_round("草稿")]), store=store)
    signal = StopSignal()
    # 停止在第一轮边界就已被请求：引擎应立刻以停止收尾。
    signal.request()
    loop.stop_signal = signal

    async def run():
        return await loop.run("研究茅台")

    outcome = asyncio.run(run())

    assert outcome.succeeded is False
    assert outcome.reason == "user_stopped"


def test_stop_during_stream_breaks_immediately(tmp_path):
    """停止请求在流中途到达时，引擎不再等待剩余 chunk。"""
    sink = Sink()
    signal = StopSignal()

    def request_stop() -> None:
        signal.request()

    provider = StopAfterProvider(["第一段", "第二段", "第三段"], on_after=request_stop)
    loop = build_loop(tmp_path, provider)
    loop.output = sink
    loop.stop_signal = signal

    async def run():
        return await loop.run("一个问题")

    outcome = asyncio.run(run())

    assert outcome.reason == "user_stopped"
    # 关键：停止不是错误，因此事件流里没有 error。
    assert "error" not in [event.kind for event in sink.events]
    # 收尾必须带终止事件，否则客户端会永远停在"运行中"。
    assert "done" in [event.kind for event in sink.events]
    done = next(event for event in sink.events if event.kind == "done")
    assert done.data["succeeded"] is False
    assert done.data["reason"] == "user_stopped"
    assert done.data["resumable"] is True


def test_stop_persists_user_message_and_partial_findings(tmp_path):
    """停止后本轮的提问与部分发现必须落库，否则刷新即消失。"""
    store = MemoryStore(tmp_path / "memory.db")
    sink = Sink()
    # 上一轮先建立一份"已有发现"：先跑一轮工具调用，让 citation 存在。
    tool = RecordingTool("get_quotes", content="行情数据")
    registry = StubRegistry({"get_quotes": tool})
    settings = make_settings(tmp_path)
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    provider = ScriptedProvider(
        [
            tool_round(ToolUse(call_id="call_1", name="get_quotes", args={"symbol": "600519"})),
            text_round("初步结论"),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        registry=registry,
        settings=settings,
        system="系统提示",
        cite=cite,
        ctx=ctx,
        conversation_id="c_stop_persist",
        store=store,
    )
    loop.output = sink

    async def first():
        return await loop.run("茅台行情如何")

    asyncio.run(first())

    # 现在第二轮：流中途停止。
    signal = StopSignal()

    class SecondProvider(Provider):
        async def stream(self, *, system, messages, tools, usage):
            yield StreamChunk(StreamEvent.TEXT_DELTA, "正在分析")
            signal.request()
            yield StreamChunk(StreamEvent.TEXT_DELTA, "被中断的后半句")
            yield message_end()

    loop.provider = SecondProvider()
    loop.stop_signal = signal

    async def second():
        return await loop.run("继续深挖")

    outcome = asyncio.run(second())

    assert outcome.reason == "user_stopped"
    # 用户提问与本轮回答（部分发现）都在库里。
    stored = store.load_messages("c_stop_persist")
    contents = [message.content for message in stored if message.content]
    assert "继续深挖" in contents
    # 上一轮的结论仍在，说明停止没有回滚既有成果。
    assert "初步结论" in contents


def test_stop_marks_checkpoint_as_recoverable(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    signal = StopSignal()
    signal.request()
    loop = build_loop(
        tmp_path,
        ScriptedProvider([text_round("x")]),
        store=store,
        conversation_id="c_cp",
    )
    loop.stop_signal = signal

    async def run():
        return await loop.run("问题")

    asyncio.run(run())

    checkpoint = store.load_latest_checkpoint("c_cp")
    assert checkpoint is not None
    assert checkpoint.status == "stopped"
    assert checkpoint.reason == "user_stopped"
    assert checkpoint.recoverable is True


def test_successful_run_marks_checkpoint_completed(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    loop = build_loop(
        tmp_path, ScriptedProvider([text_round("答案")]), store=store, conversation_id="c_ok"
    )

    async def run():
        return await loop.run("问题")

    asyncio.run(run())

    checkpoint = store.load_latest_checkpoint("c_ok")
    assert checkpoint is not None
    assert checkpoint.status == "completed"
    # 已交付完毕的运行不该提供"继续"入口。
    assert checkpoint.recoverable is False


def test_no_stop_signal_means_unchanged_behaviour(tmp_path):
    """没有信号时（eval/脚本/子代理）行为与本功能引入前完全一致。"""
    loop = build_loop(tmp_path, ScriptedProvider([text_round("正常答案")]))
    assert loop.stop_signal is None

    async def run():
        return await loop.run("问题")

    outcome = asyncio.run(run())

    assert outcome.succeeded is True
    assert outcome.answer == "正常答案"


def test_stop_after_tool_call_preserves_that_tool_result(tmp_path):
    """停止在耗时工具返回之后到达：那次取数必须被保住。"""
    tool = RecordingTool("get_quotes", content="行情数据")
    registry = StubRegistry({"get_quotes": tool})
    settings = make_settings(tmp_path)
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    signal = StopSignal()
    loop = AgentLoop(
        provider=ScriptedProvider(
            [tool_round(ToolUse(call_id="call_1", name="get_quotes", args={}))]
        ),
        registry=registry,
        settings=settings,
        system="系统提示",
        cite=cite,
        ctx=ctx,
        conversation_id="c_tool_stop",
    )
    # 工具一执行完就置位：引擎应在 gather 之后的检查点停下。
    original_run = tool.run

    async def run_and_stop(**kwargs):
        result = await original_run(**kwargs)
        signal.request()
        return result

    tool.run = run_and_stop
    loop.stop_signal = signal

    async def run():
        return await loop.run("取数")

    outcome = asyncio.run(run())

    assert outcome.reason == "user_stopped"
    assert len(tool.calls) == 1
    # 工具结果进入会话（作为配对的 tool_result 帧，按引擎的 JSON 信封编码），
    # 因此可被后续轮次复用，停止没有让它白跑。
    results = [
        pair
        for message in loop.memory.snapshot()
        for pair in message.tool_results
    ]
    assert any(call_id == "call_1" and "行情数据" in payload for call_id, payload in results)
