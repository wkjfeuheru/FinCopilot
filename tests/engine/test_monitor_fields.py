"""监控所需的引擎事件字段：拒绝 verdict 与事件轮次序号（docs 03.14.4）。

这些字段是监控指标的口径来源——"安全拦截"与"工具失败"靠 ``verdict``
区分，"哪一步开始跑偏"靠 ``turn`` 定位。它们必须稳定存在，否则指标会
把治理拦截误算成工具脆弱。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from test_loop import (  # noqa: E402
    RecordingTool,
    ScriptedProvider,
    Sink,
    StubRegistry,
    make_loop,
    text_round,
    tool_round,
)

from finharness.config.settings import ContextSettings, Settings  # noqa: E402
from finharness.permissions.gate import PermissionLevel  # noqa: E402
from finharness.types import ToolUse  # noqa: E402


def _settings(*, max_turns: int = 6, max_identical_tool_calls: int = 3) -> Settings:
    return Settings(
        context=ContextSettings(
            max_turns=max_turns, max_identical_tool_calls=max_identical_tool_calls
        )
    )


def _failed_statuses(events):
    return [
        event.data
        for event in events
        if event.kind == "tool_status" and event.data.get("status") == "failed"
    ]


def test_denied_tool_carries_denied_verdict():
    """权限门拒绝（无确认通道的写操作）应带 verdict=denied。"""
    sink = Sink()
    write_tool = RecordingTool("write_file", permission=PermissionLevel.WRITE)
    provider = ScriptedProvider(
        [tool_round(ToolUse("c1", "write_file", {"path": "output/x.md", "content": "hi"})), text_round("好的")]
    )
    loop = make_loop(
        provider,
        registry=StubRegistry({"write_file": write_tool}),
        settings=_settings(),
        output=sink,
    )
    asyncio.run(loop.run("写个文件"))
    failed = _failed_statuses(sink.events)
    assert any(s.get("verdict") == "denied" for s in failed), failed


def test_unknown_tool_carries_unknown_verdict():
    """未知工具应带 verdict=unknown（结构性拒绝，而非执行失败）。"""
    sink = Sink()
    provider = ScriptedProvider(
        [tool_round(ToolUse("c1", "no_such_tool", {})), text_round("好的")]
    )
    loop = make_loop(
        provider,
        registry=StubRegistry({}),
        settings=_settings(),
        output=sink,
    )
    asyncio.run(loop.run("调用不存在的工具"))
    failed = _failed_statuses(sink.events)
    assert failed and failed[0].get("verdict") == "unknown"


def test_loop_guard_event_carries_turn():
    """重复调用拦截事件应带轮次序号，使"哪一步跑偏"可定位。"""
    sink = Sink()
    args = {"symbol": "600519"}
    tool = RecordingTool("get_quote", content="报价")
    provider = ScriptedProvider(
        [tool_round(ToolUse(f"c{i}", "get_quote", dict(args))) for i in range(4)]
        + [text_round("最终回答")]
    )
    loop = make_loop(
        provider,
        registry=StubRegistry({"get_quote": tool}),
        settings=_settings(max_turns=10, max_identical_tool_calls=2),
        output=sink,
    )
    asyncio.run(loop.run("反复查同一只股票"))
    guards = [event.data for event in sink.events if event.kind == "loop_guard"]
    assert guards, [e.kind for e in sink.events]
    assert all("turn" in g for g in guards)
    assert all(isinstance(g["turn"], int) for g in guards)
