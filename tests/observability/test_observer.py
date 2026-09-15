"""观测门面：降级、错误分类与 Span 行为（docs 03.14）。"""

import asyncio
import logging

from finharness.observability import NullObserver, Observer, build_observer
from finharness.observability.metrics import MetricsRecorder
from finharness.observability.observer import classify_error
from finharness.provider.errors import (
    AuthError,
    NetworkError,
    RateLimitError,
    ServerError,
    TokenLimitError,
)


class _Usage:
    input_tokens = 10
    output_tokens = 5
    cache_hit_tokens = 2
    cache_miss_tokens = 0


class _RecordingTracer:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.finished: list[dict] = []

    def start_run(self, *, name, run_type, inputs, parent):
        handle = {"name": name, "run_type": run_type, "parent": parent}
        self.started.append(handle)
        return handle

    def finish_run(self, handle, *, attributes, outputs=None, error=None) -> None:
        self.finished.append({"name": handle["name"], "error": error, "outputs": outputs})


def test_error_classification_maps_provider_errors() -> None:
    assert classify_error(TokenLimitError("too long")) == "token_limit"
    assert classify_error(AuthError("nope")) == "auth_error"
    assert classify_error(RateLimitError("slow down")) == "rate_limit"
    assert classify_error(ServerError("boom")) == "server_error"
    assert classify_error(TimeoutError("timed out")) == "llm_timeout"


def test_network_error_with_timeout_message_is_llm_timeout() -> None:
    assert classify_error(NetworkError("Provider timed out waiting for first byte")) == "llm_timeout"
    assert classify_error(NetworkError("connection reset")) == "network_error"


def test_null_observer_is_a_silent_noop() -> None:
    observer = NullObserver()
    assert observer.enabled is False
    assert observer.bind_request(session_id="s") == ""

    async def main() -> None:
        with observer.request_span(message="hi") as span:
            span.set_attribute("x", 1)
        async with observer.llm_span(model="m") as span:
            span.mark_first_token()
        async with observer.tool_span(name="t") as span:
            observer.tool_finished(tool="t", status="ok", span=span)
        observer.record_error(error_type="x")

    asyncio.run(main())


def test_span_records_usage_and_first_token_on_tracer() -> None:
    tracer = _RecordingTracer()
    observer = Observer(tracer=tracer)

    async def main() -> None:
        async with observer.llm_span(model="deepseek-chat", call_type="main") as span:
            span.mark_first_token()
            span.set_usage(_Usage())

    asyncio.run(main())

    assert [run["name"] for run in tracer.started] == ["llm.main"]
    finished = tracer.finished[0]
    assert finished["outputs"]["input_tokens"] == 10
    assert finished["outputs"]["output_tokens"] == 5


def test_governed_tool_statuses_do_not_count_as_tool_failure() -> None:
    """治理拦截不应污染"哪个工具最脆弱"的统计。"""
    metrics = MetricsRecorder()
    observer = Observer(metrics=metrics)

    async def main() -> None:
        for status in ("ok", "denied", "blocked"):
            async with observer.tool_span(name="get_quote") as span:
                span.set_attribute("status", status)
                observer.tool_finished(tool="get_quote", status=status, span=span)

    asyncio.run(main())

    rendered = metrics.render().decode()
    assert "tool_failure" not in rendered
    assert 'agent_tool_calls_total{status="denied",tool="get_quote"} 1.0' in rendered


def test_backend_exception_never_breaks_the_caller() -> None:
    """观测后端抛错时，引擎调用方不应感知——这是审计 hook 的同一原则。"""

    class ExplodingMetrics:
        def request_finished(self, **kwargs):
            raise RuntimeError("backend down")

        def llm_finished(self, **kwargs):
            raise RuntimeError("backend down")

        def tool_finished(self, **kwargs):
            raise RuntimeError("backend down")

        def error(self, **kwargs):
            raise RuntimeError("backend down")

    observer = Observer(metrics=ExplodingMetrics(), logger=logging.getLogger("finharness.test.exploding"))

    async def main() -> None:
        with observer.request_span(message="hi"):
            pass
        async with observer.llm_span(model="m"):
            pass
        async with observer.tool_span(name="t") as span:
            observer.tool_finished(tool="t", status="error", span=span)

    asyncio.run(main())


def test_build_observer_disabled_returns_null() -> None:
    class Section:
        class Logging:
            level = "INFO"
            capture_payloads = False
            max_payload_chars = 100

        class Metrics:
            enabled = False

        class Tracing:
            enabled = False
            capture_payloads = False

        logging = Logging()
        metrics = Metrics()
        tracing = Tracing()

    class Settings:
        observability = Section()

    observer = build_observer(Settings())

    assert isinstance(observer, NullObserver)
