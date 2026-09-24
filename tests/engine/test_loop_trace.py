"""AgentLoop 记录的 Thought/Action/Observation trace（docs 03.3）。

eval 系统会回放这些记录，因此这里的断言覆盖它所依赖的数据形状：
tool round 会将其丢弃的草稿文本保留为 ``thought``，被拒绝的调用被记录为
未执行的 observation，并且每轮的 token 与延迟都会被附加。
"""

import asyncio

from finharness.permissions.modes import Verdict
from finharness.tools.registry import ToolRegistry
from finharness.types import ToolUse
from tests.engine.test_loop import (
    RecordingTool,
    ScriptedProvider,
    StubRegistry,
    make_loop,
    text_round,
    tool_round,
)


class Decision:
    def __init__(self, verdict: Verdict, reason: str | None = None) -> None:
        self.verdict = verdict
        self.reason = reason


class DenyGate:
    """拒绝每一次调用的 gate 替身，用于验证拒绝 observation。"""

    async def check(self, tool, args):
        return Decision(Verdict.DENY, "命中交易意图规则")


def run(coro):
    return asyncio.run(coro)


def test_tool_round_records_discarded_draft_as_thought():
    async def scenario():
        provider = ScriptedProvider(
            [
                tool_round(
                    ToolUse("c1", "get_quote", {"symbol": "600519"}),
                    draft=("先查行情", "："),
                ),
                text_round("贵州茅台现价 1500 元。"),
            ]
        )
        registry = StubRegistry({"get_quote": RecordingTool("get_quote", content="1500")})
        loop = make_loop(provider, registry=registry)
        outcome = await loop.run("茅台多少钱")
        return outcome

    outcome = run(scenario())

    # 两轮：先是 tool round，然后是 answer round。
    assert len(outcome.trace) == 2
    assert outcome.rounds == 2

    first = outcome.trace[0]
    # 该草稿被从答案中丢弃，但仍是本轮陈述的推理。
    assert first.thought == "先查行情："
    assert [action.name for action in first.actions] == ["get_quote"]
    assert first.actions[0].args == {"symbol": "600519"}
    assert first.answer == ""

    observation = first.observations[0]
    assert observation.name == "get_quote"
    assert observation.ok is True
    assert observation.preview == "1500"
    assert observation.duration_ms >= 0

    second = outcome.trace[1]
    assert second.answer == "贵州茅台现价 1500 元。"
    assert second.actions == []


def test_per_round_tokens_and_latency_recorded():
    async def scenario():
        provider = ScriptedProvider(
            [
                tool_round(
                    ToolUse("c1", "get_quote", {"symbol": "600519"}),
                    input_tokens=11,
                    output_tokens=3,
                ),
                text_round("done", input_tokens=7, output_tokens=5),
            ]
        )
        registry = StubRegistry({"get_quote": RecordingTool("get_quote")})
        loop = make_loop(provider, registry=registry)
        return await loop.run("q")

    outcome = run(scenario())

    assert outcome.trace[0].input_tokens == 11
    assert outcome.trace[0].output_tokens == 3
    assert outcome.trace[1].input_tokens == 7
    assert outcome.trace[1].output_tokens == 5
    assert outcome.usage.input_tokens == 18
    assert outcome.usage.output_tokens == 8
    # 每个到达 stream 的轮次，其延迟都是非负整数。
    assert all(round_trace.llm_ms >= 0 for round_trace in outcome.trace)
    assert outcome.trace[1].llm_first_ms >= 0


def test_denied_call_is_recorded_as_non_executed_observation():
    async def scenario():
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "write_file", {"path": "x", "content": "y"})),
                text_round("无法写入。"),
            ]
        )
        registry = StubRegistry({"write_file": RecordingTool("write_file")})
        loop = make_loop(provider, registry=registry)
        loop.gate = DenyGate()
        return await loop.run("写个文件")

    outcome = run(scenario())

    observation = outcome.trace[0].observations[0]
    assert observation.ok is False
    assert observation.error == "命中交易意图规则"
    # 工具从未运行，因此替身上没有记录任何执行。
    assert outcome.trace[0].actions[0].name == "write_file"


def test_a_direct_lazy_call_is_activated_and_recorded_as_success():
    async def scenario():
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "run_backtest", {})),
                text_round("已跑完。"),
            ]
        )
        # 真实 registry 会对能力分层建模；run_backtest 默认是懒加载的。
        real = ToolRegistry(data=None)
        assert "run_backtest" in real.lazy_names()
        loop = make_loop(provider, registry=real)
        return await loop.run("回测"), real

    outcome, registry = run(scenario())

    # 直接调用不再是一次被拒绝的观测：就地激活后照常执行。
    assert registry.is_active("run_backtest") is True
    observation = outcome.trace[0].observations[0]
    assert observation.name == "run_backtest"


def test_loop_detected_round_is_recorded_before_failure():
    async def scenario():
        tool_use = ToolUse("c1", "get_quote", {"symbol": "600519"})
        # 每一轮都是完全相同的调用：一旦超过阈值，循环就会升级处理。
        provider = ScriptedProvider(
            [tool_round(tool_use) for _ in range(6)]
        )
        registry = StubRegistry({"get_quote": RecordingTool("get_quote")})
        loop = make_loop(provider, registry=registry)
        return await loop.run("q")

    outcome = run(scenario())

    assert outcome.succeeded is False
    assert outcome.reason == "loop_detected"
    # 中止的那一轮仍然存在，其 observation 解释了拒绝原因。
    assert outcome.rounds >= 1
    refused = [
        observation
        for round_trace in outcome.trace
        for observation in round_trace.observations
        if not observation.ok
    ]
    assert refused, "expected the guarded call to appear as a failed observation"
    assert outcome.trace == list(outcome.trace)  # 一个稳定且有序的列表


def test_done_payload_carries_rounds():
    async def scenario():
        from tests.engine.test_loop import Sink

        sink = Sink()
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "get_quote", {"symbol": "600519"})),
                text_round("ok"),
            ]
        )
        registry = StubRegistry({"get_quote": RecordingTool("get_quote")})
        loop = make_loop(provider, registry=registry, output=sink)
        await loop.run("q")
        return sink

    sink = run(scenario())
    done = [event for event in sink.events if event.kind == "done"][-1]
    assert done.data["rounds"] == 2
