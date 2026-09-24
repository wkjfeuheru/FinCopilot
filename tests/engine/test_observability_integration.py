"""可观测性插桩：引擎主循环的 Span 层级、工具状态与 call_type（docs 03.14）。"""

import asyncio

from finharness.observability import NullObserver
from finharness.observability.observer import Observer
from finharness.types import ModelUsage, StreamChunk, StreamEvent, ToolUse
from tests.engine.test_loop import (  # noqa: E402 - 复用既有替身
    RecordingTool,
    ScriptedProvider,
    StubRegistry,
    make_loop,
    text_round,
    tool_round,
)


class RecordingTracer:
    """记录 Span 的开始/结束，用于断言层级与属性。"""

    def __init__(self) -> None:
        self.started: list[dict] = []
        self.finished: list[dict] = []

    def start_run(self, *, name, run_type, inputs, parent):
        handle = {"name": name, "run_type": run_type, "parent": parent, "inputs": inputs}
        self.started.append(handle)
        return handle

    def finish_run(self, handle, *, attributes, outputs=None, error=None) -> None:
        self.finished.append(
            {"name": handle["name"], "parent": handle["parent"], "error": error}
        )


class RecordingMetrics:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.llms: list[dict] = []
        self.tools: list[dict] = []
        self.errors: list[str] = []

    def request_finished(self, **kwargs):
        self.requests.append(kwargs)

    def llm_finished(self, **kwargs):
        self.llms.append(kwargs)

    def tool_finished(self, **kwargs):
        self.tools.append(kwargs)

    def error(self, **kwargs):
        self.errors.append(kwargs["error_type"])


def test_llm_span_wraps_the_consumed_stream_and_carries_usage() -> None:
    """Span 必须包住整个 async for 消费过程，并带上真实 token 用量。"""
    tracer = RecordingTracer()
    metrics = RecordingMetrics()
    observer = Observer(metrics=metrics, tracer=tracer)
    provider = ScriptedProvider([text_round("answer", input_tokens=7, output_tokens=3)])

    loop = make_loop(provider, observer=observer)
    asyncio.run(loop.run("question"))

    llm_runs = [run for run in tracer.started if run["run_type"] == "llm"]
    assert len(llm_runs) == 1
    assert llm_runs[0]["name"] == "llm.main"
    assert llm_runs[0]["parent"] is not None  # 挂在请求根 Span 之下
    assert metrics.llms[0]["input_tokens"] == 7
    assert metrics.llms[0]["output_tokens"] == 3
    assert metrics.llms[0]["call_type"] == "main"
    assert metrics.llms[0]["first_token_s"] is not None


def test_request_span_reports_status_and_reason() -> None:
    tracer = RecordingTracer()
    observer = Observer(tracer=tracer)
    loop = make_loop(ScriptedProvider([text_round("ok")]), observer=observer)

    asyncio.run(loop.run("question"))

    roots = [run for run in tracer.finished if run["name"] == "finharness.turn"]
    assert len(roots) == 1


def test_tool_span_records_status_and_failure_is_counted() -> None:
    metrics = RecordingMetrics()
    observer = Observer(metrics=metrics)
    registry = StubRegistry({"get_quote": RecordingTool("get_quote", ok=False, error="upstream")})
    provider = ScriptedProvider(
        [
            tool_round(ToolUse(call_id="c1", name="get_quote", args={"symbol": "600519"})),
            text_round("done"),
        ]
    )
    loop = make_loop(provider, registry=registry, observer=observer)

    asyncio.run(loop.run("question"))

    statuses = [call["status"] for call in metrics.tools]
    assert "error" in statuses
    assert "tool_failure" in metrics.errors


def test_denied_tool_is_tagged_but_not_counted_as_failure() -> None:
    """权限拒绝是治理结果，不是工具脆弱。"""
    metrics = RecordingMetrics()

    class DenyAll:
        def __init__(self):
            from finharness.permissions.modes import Verdict

            self._verdict = Verdict.DENY

        async def check(self, tool, args):
            from finharness.permissions.gate import GateDecision

            return GateDecision(verdict=self._verdict, reason="blocked by policy")

    observer = Observer(metrics=metrics)
    registry = StubRegistry({"get_quote": RecordingTool("get_quote")})
    provider = ScriptedProvider(
        [
            tool_round(ToolUse(call_id="c1", name="get_quote", args={"symbol": "600519"})),
            text_round("done"),
        ]
    )
    loop = make_loop(provider, registry=registry, observer=observer)
    loop.gate = DenyAll()

    asyncio.run(loop.run("question"))

    assert any(call["status"] == "denied" for call in metrics.tools)
    assert "tool_failure" not in metrics.errors


def test_null_observer_leaves_the_loop_unchanged() -> None:
    """不注入 observer（默认）时，行为与引入观测之前一致。"""
    provider = ScriptedProvider([text_round("plain")])
    loop = make_loop(provider)

    assert isinstance(loop.observer, NullObserver)
    outcome = asyncio.run(loop.run("question"))

    assert outcome.succeeded
    assert outcome.answer == "plain"


def test_retry_callback_records_stat_and_log() -> None:
    """重试既计入会话统计，也应产生一条可排查日志。"""
    from finharness.engine.retry import RetryPolicy
    from finharness.provider.errors import RateLimitError

    class FlakyOnce:
        def __init__(self):
            self.calls = 0

        async def stream(self, *, system, messages, tools, usage):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("slow down", retry_after_s=0.0)
            yield StreamChunk(StreamEvent.TEXT_DELTA, "recovered")
            yield StreamChunk(
                StreamEvent.MESSAGE_END, ModelUsage(input_tokens=1, output_tokens=1)
            )

    loop = make_loop(
        FlakyOnce(),
        retry_policy=RetryPolicy(max_retries=3, base_delay_s=0.0, cap_delay_s=0.0),
    )
    outcome = asyncio.run(loop.run("question"))

    assert outcome.succeeded
    assert outcome.retry_count == 1
