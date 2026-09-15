"""计划进度与偏离信号 (docs 03.3.9)。

这些信号是建议性的：每个 round 都会带着它的进度被发出，并且当它
看起来偏离计划、或计划停止推进时，会有一个软提示附加到下一次
请求上。任何调用都不会因为偏离计划而被拒绝，因此测试断言的是事件
和注入的文本，而不是结果。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from test_loop import (  # noqa: E402
    RecordingTool,
    ScriptedProvider,
    Sink,
    StubRegistry,
    make_loop,
    make_settings,
    text_round,
    tool_round,
)

from finharness.context.session import PlanStep  # noqa: E402
from finharness.types import ToolUse  # noqa: E402


def build_loop(provider, *, plan_steps, tools, sink, max_turns=30):
    loop = make_loop(
        provider,
        registry=StubRegistry(tools),
        settings=make_settings(max_turns=max_turns),
        output=sink,
    )
    if plan_steps is not None:
        loop.ctx.set_plan("研究任务", plan_steps)
    return loop


def plan_progress_events(events):
    return [event.data for event in events if event.kind == "plan_progress"]


def test_on_plan_round_reports_progress_without_drift():
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("call_1", "get_quote", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="取 600519 行情", tool_hint=["get_quote"])],
            tools={"get_quote": tool},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())
    progress = plan_progress_events(events)

    assert len(progress) == 1
    assert progress[0]["done"] == 0
    assert progress[0]["total"] == 1
    assert progress[0]["drift"] == []
    assert progress[0]["revision"] == 1


def test_a_tool_off_the_hint_list_is_not_a_deviation():
    """取数、读数据、画图——任何工具都是达成任务的手段。使用了
    计划恰好未提及的工具，并不是偏离目标。"""
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_kline", content="K线")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("call_1", "get_kline", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="取 600519 行情", tool_hint=["get_quote"])],
            tools={"get_kline": tool},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())

    # get_kline 未被提示，但它针对的是任务所指定的同一标的。
    assert plan_progress_events(events)[0]["drift"] == []


def test_a_symbol_outside_the_task_scope_is_reported_as_drift():
    """真正的偏离：分析了任务从未询问过的标的。"""
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("call_1", "get_quote", {"symbol": "600036"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="分析 600519", tool_hint=["get_quote"])],
            tools={"get_quote": tool},
            sink=sink,
        )
        outcome = await loop.run("问题")
        return outcome, sink.events

    outcome, events = asyncio.run(run())
    progress = plan_progress_events(events)

    assert progress[0]["drift"] == ["600036"]
    # 调用仍然执行了：偏离是一种信号，而不是门控。
    assert outcome.succeeded is True


def test_drift_prompts_once_per_symbol():
    """已上报过的越界标的不会在每个 round 重复提示。"""
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_quote", {"symbol": "600036"})),
                    tool_round(ToolUse("c2", "get_quote", {"symbol": "600036"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="分析 600519", tool_hint=["get_quote"])],
            tools={"get_quote": tool},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())
    drift = [item["drift"] for item in plan_progress_events(events)]

    assert drift == [["600036"], []]


def test_a_plan_without_a_named_target_never_reports_drift():
    """行业/宏观类任务不指定标的，因此没有标的会偏离目标。"""
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("call_1", "get_quote", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="分析白酒行业", tool_hint=["get_quote"])],
            tools={"get_quote": tool},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events)[0]["drift"] == []


def test_drift_prompt_is_attached_to_the_next_request():
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_quote", {"symbol": "600036"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="分析 600519", tool_hint=["get_quote"])],
            tools={"get_quote": tool},
            sink=sink,
        )
        await loop.run("问题")
        return loop

    loop = asyncio.run(run())

    assert "目标核对" in loop._state_text()
    assert "600036" in loop._state_text()


def test_stalled_plan_is_counted_and_prompts_a_revision():
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("call_1", "get_quote", {"symbol": "600519"})),
                    tool_round(ToolUse("call_2", "get_quote", {"symbol": "600519"})),
                    tool_round(ToolUse("call_3", "get_quote", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="取 600519 行情", tool_hint=["get_quote"])],
            tools={"get_quote": tool},
            sink=sink,
        )
        await loop.run("问题")
        return loop, sink.events

    loop, events = asyncio.run(run())
    stalls = [item["stalled_turns"] for item in plan_progress_events(events)]

    # 计划的状态从不改变，因此停滞计数每个 round 都在攀升。
    assert stalls == [0, 1, 2]
    # 超过阈值后，下一次请求会带上一条软性停滞提示。
    assert "停滞提示" in loop._state_text()


# --- 能力不匹配（任务用错了工具） -------------------------------------------

def test_using_a_different_capability_than_the_plan_hints_is_reported():
    """典型的用错工具案例：任务关于公告，却用新闻来回答。
    调用的 *形态* 相同，能力不同。"""
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_market_news", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[
                PlanStep(seq=1, action="查公告", tool_hint=["get_announcements"])
            ],
            tools={"get_market_news": RecordingTool("get_market_news")},
            sink=sink,
        )
        await loop.run("问题")
        return loop, sink.events

    loop, events = asyncio.run(run())
    progress = plan_progress_events(events)

    assert progress[0]["mismatch"] == ["新闻"]
    # 软信号：调用仍然执行了，提示会附加到下一次请求。
    assert "工具核对" in loop._state_text()
    assert "新闻" in loop._state_text()


def test_a_sibling_tool_of_the_same_capability_is_not_a_mismatch():
    """quote 与 kline 都属于 "price"：用一个替换另一个是
    策略选择，而不是用错工具。"""
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_kline", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="取行情", tool_hint=["get_quote"])],
            tools={"get_kline": RecordingTool("get_kline")},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events)[0]["mismatch"] == []


def test_presentation_and_process_tools_never_mismatch():
    """make_chart / load_skill 负责工作如何呈现与路由，而不是
    使用哪些证据——无论提示如何，它们都豁免。"""
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(
                        ToolUse("c1", "make_chart", {"title": "图"}),
                        ToolUse("c2", "load_skill", {"name": "equity-research"}),
                    ),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="查公告", tool_hint=["get_announcements"])],
            tools={
                "make_chart": RecordingTool("make_chart"),
                "load_skill": RecordingTool("load_skill"),
            },
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events)[0]["mismatch"] == []


def test_a_plan_without_tool_hints_never_reports_mismatch():
    """未声明任何提示 = 未表达任何期望，因此没有什么会是错的。"""
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_market_news", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="了解公司近况")],  # 无 tool_hint
            tools={"get_market_news": RecordingTool("get_market_news")},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events)[0]["mismatch"] == []


def test_mismatch_is_reported_once_per_capability():
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_market_news", {"symbol": "600519"})),
                    tool_round(ToolUse("c2", "get_market_news", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="查公告", tool_hint=["get_announcements"])],
            tools={"get_market_news": RecordingTool("get_market_news")},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())
    mismatches = [item["mismatch"] for item in plan_progress_events(events)]

    assert mismatches == [["新闻"], []]


def test_no_plan_emits_no_progress_events():
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("call_1", "get_quote", {})),
                    text_round("答案"),
                ]
            ),
            plan_steps=None,
            tools={"get_quote": tool},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events) == []


def test_done_payload_exposes_the_plan():
    import asyncio

    async def run():
        sink = Sink()
        tool = RecordingTool("get_quote", content="报价")
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("call_1", "get_quote", {})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="取行情", tool_hint=["get_quote"])],
            tools={"get_quote": tool},
            sink=sink,
        )
        await loop.run("问题")
        return sink.events

    events = asyncio.run(run())
    done = next(event for event in events if event.kind == "done")

    assert done.data["plan"] == {"revision": 1, "done": 0, "total": 1}


# --- 基于意图的范围判定（误报修复） -----------------------------------------

def test_plan_hint_symbol_is_not_treated_as_task_scope():
    """仅出现在工具提示中的标的不是研究目标。

    真实案例：某个计划步骤在提示 get_peers 时携带了一个标的，这使得
    计划看起来像是把任务范围限定在了那一家公司。
    """
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_peers", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="了解行业格局", tool_hint=["get_peers"])],
            tools={"get_peers": RecordingTool("get_peers")},
            sink=sink,
        )
        await loop.run("分析白酒行业竞争格局")
        return sink.events

    events = asyncio.run(run())

    # 问题或计划正文中都没有标的 ⇒ 没有任何越界。
    assert plan_progress_events(events)[0]["drift"] == []


def test_the_questions_own_symbol_defines_scope_even_if_the_plan_omits_it():
    """对于单一标的的任务，问题就是事实依据：使用它的标的
    即在范围内，无论计划如何表述。"""
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_indicators", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="分析盈利能力", tool_hint=["get_indicators"])],
            tools={"get_indicators": RecordingTool("get_indicators")},
            sink=sink,
        )
        await loop.run("贵州茅台(600519)的盈利能力怎么样？")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events)[0]["drift"] == []


def test_a_capability_implied_by_the_plan_prose_is_not_a_mismatch():
    """计划正文中隐含的能力不算不匹配：哪怕某个步骤的提示列表
    忘记列出估值工具，只要它写着 "分析估值" 就需要估值能力——看意图，而不是记账。"""
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_valuation", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[
                PlanStep(seq=1, action="分析贵州茅台估值水平", tool_hint=["get_indicators"])
            ],
            tools={"get_valuation": RecordingTool("get_valuation")},
            sink=sink,
        )
        await loop.run("分析茅台")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events)[0]["mismatch"] == []


def test_a_capability_named_by_the_question_is_not_a_mismatch():
    """任务自身的措辞也算数：关于盈利的问题需要财务
    数据，即使计划描述得不够充分。"""
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_financials", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="第一步", tool_hint=["get_quote"])],
            tools={"get_financials": RecordingTool("get_financials")},
            sink=sink,
        )
        await loop.run("分析贵州茅台的盈利质量和财务表现")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events)[0]["mismatch"] == []


def test_a_truly_unrelated_capability_still_reports():
    """防止过度抑制：关于估值的问题与计划，仍应标记出
    无人要求过的公告抓取。"""
    import asyncio

    async def run():
        sink = Sink()
        loop = build_loop(
            ScriptedProvider(
                [
                    tool_round(ToolUse("c1", "get_announcements", {"symbol": "600519"})),
                    text_round("答案"),
                ]
            ),
            plan_steps=[PlanStep(seq=1, action="分析估值水平", tool_hint=["get_valuation"])],
            tools={"get_announcements": RecordingTool("get_announcements")},
            sink=sink,
        )
        await loop.run("贵州茅台(600519)估值贵不贵？")
        return sink.events

    events = asyncio.run(run())

    assert plan_progress_events(events)[0]["mismatch"] == ["公告"]
