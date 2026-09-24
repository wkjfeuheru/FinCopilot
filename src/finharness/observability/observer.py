"""观测门面：调用方只依赖 ``Observer``，三个后端各自独立可降级（docs 03.14）。

设计要点：

* **尽力而为**。与 ``HookChain.post`` 同一哲学——观测绝不能破坏一次回合，
  因此每个后端调用都包在 ``_guard`` 里，失败只降级不抛出。
* **深模块**。引擎只看到 ``request_span``/``llm_span``/``tool_span`` 三个
  上下文管理器与 ``Span`` 句柄；是否接 Prometheus、是否接 LangSmith 对
  调用方透明。
* **后端缺席即 no-op**。``NullObserver`` 让不关心观测的调用方（测试替身、
  ``eval/``）无需改动。
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Protocol, runtime_checkable

from finharness.observability.context import TraceContext, bind_trace, current_trace

__all__ = [
    "NullObserver",
    "Observer",
    "Span",
    "build_observer",
    "classify_error",
    "current_trace_id",
]

_log = logging.getLogger("finharness.observability")

# 当前 Span 对应的后端 run 句柄栈，使嵌套 Span 能拿到正确的 parent，
# 且无需调用方显式传递。用 ContextVar 保证并发工具调用各自独立。
_RUN_STACK: ContextVar[tuple[Any, ...]] = ContextVar("finharness_run_stack", default=())


# 治理拦截状态：出现在 agent_tool_calls_total 里，但不是工具失败。
_GOVERNED_STATUSES = frozenset({"ok", "denied", "blocked", "loop_guard"})


@runtime_checkable
class _MetricsSink(Protocol):
    """Prometheus 后端的接口（见 ``metrics.py``）。"""

    def request_finished(self, *, status: str, duration_s: float) -> None: ...

    def llm_finished(
        self,
        *,
        model: str,
        call_type: str,
        duration_s: float,
        first_token_s: float | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> None: ...

    def tool_finished(self, *, tool: str, status: str, duration_s: float) -> None: ...

    def error(self, *, error_type: str) -> None: ...


@runtime_checkable
class _Tracer(Protocol):
    """追踪后端的接口（见 ``tracing.py``）。"""

    def start_run(
        self, *, name: str, run_type: str, inputs: dict, parent: Any | None
    ) -> Any | None: ...

    def finish_run(
        self,
        handle: Any,
        *,
        attributes: dict,
        outputs: dict | None = None,
        error: str | None = None,
    ) -> None: ...


def current_trace_id() -> str:
    """当前上下文的 trace id；未绑定时为空串。"""
    trace = current_trace()
    return trace.trace_id if trace else ""


def _guard(action: str) -> None:
    """观测失败只告警，绝不向引擎抛出。"""
    _log.debug("observability %s failed", action, exc_info=True)


def classify_error(exc: BaseException) -> str:
    """把异常归类为指标标签；用于 ``agent_request_errors_total``。

    导入放在函数内，避免 ``observability`` 与 ``provider`` 在模块加载期互相
    牵扯。
    """

    try:
        from finharness.provider.errors import (
            AuthError,
            NetworkError,
            ProviderError,
            RateLimitError,
            ServerError,
            TokenLimitError,
        )
    except Exception:  # noqa: BLE001 - provider 层缺席时只按名字判断
        return type(exc).__name__

    if isinstance(exc, TokenLimitError):
        return "token_limit"
    if isinstance(exc, AuthError):
        return "auth_error"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, ServerError):
        return "server_error"
    if isinstance(exc, NetworkError):
        message = str(exc).lower()
        if "timeout" in message or "timed out" in message:
            return "llm_timeout"
        return "network_error"
    if isinstance(exc, ProviderError):
        return "provider_error"
    if isinstance(exc, TimeoutError):
        return "llm_timeout"
    return type(exc).__name__


class Span:
    """一次操作的观测句柄；后端缺席时退化为廉价 no-op。

    ``set_usage`` 让 LLM Span 在结束时一次性发出 token 指标，避免调用方
    到处记数、也避免子 Agent 与主循环双计。
    """

    __slots__ = ("_observer", "_handle", "_started", "_first_token_at", "_attrs", "_usage", "_ended", "_error")

    def __init__(self, observer: "Observer", handle: Any | None = None) -> None:
        self._observer = observer
        self._handle = handle
        self._started = time.monotonic()
        self._first_token_at: float | None = None
        self._attrs: dict[str, Any] = {}
        self._usage: Any | None = None
        self._ended = False
        self._error: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self._attrs[key] = value

    def mark_first_token(self) -> None:
        """记录首个 token 到达的时刻（用于首 token 延迟）。"""
        if self._first_token_at is None:
            self._first_token_at = time.monotonic()

    def set_usage(self, usage: Any) -> None:
        """记录本次 LLM 调用的 token 用量，随 Span 结束时上报。"""
        self._usage = usage

    @property
    def attributes(self) -> dict[str, Any]:
        return dict(self._attrs)

    def duration_s(self) -> float:
        return time.monotonic() - self._started

    def first_token_s(self) -> float | None:
        if self._first_token_at is None:
            return None
        return self._first_token_at - self._started

    def fail(self, error: str) -> None:
        """标记本次操作失败，供追踪后端记录错误状态。"""
        self._error = error


class NullObserver:
    """空实现：不接任何后端，所有方法都是 no-op。"""

    enabled = False

    def bind_request(
        self, *, session_id: str = "", conversation_id: str = ""
    ) -> str:
        return ""

    @contextmanager
    def request_span(
        self, *, message: str = "", mode: str = "", model: str = "", emit_metric: bool = True
    ) -> Iterator[Span]:
        yield Span(self)  # type: ignore[arg-type]

    @asynccontextmanager
    async def llm_span(
        self, *, model: str, call_type: str = "main", messages: Any = None, tool_count: int = 0
    ):
        yield Span(self)  # type: ignore[arg-type]

    @asynccontextmanager
    async def tool_span(self, *, name: str, args: Any = None, call_id: str = ""):
        yield Span(self)  # type: ignore[arg-type]

    def tool_finished(self, *, tool: str, status: str, span: Any = None) -> None:
        return None

    def log_tool_call(self, *, tool: str, args: Any, turn: int = 0) -> None:
        return None

    def record_error(self, *, error_type: str, detail: str = "") -> None:
        return None

    def record_governance_event(self, *, kind: str) -> None:
        return None


class Observer:
    """真实实现：组合可选的指标与追踪后端，并发射结构化日志。"""

    enabled = True

    def __init__(
        self,
        *,
        metrics: Any | None = None,
        tracer: Any | None = None,
        logger: logging.Logger | None = None,
        capture_payloads: bool = False,
        tracing_capture_payloads: bool = False,
        max_payload_chars: int = 4000,
    ) -> None:
        self.metrics = metrics
        self.tracer = tracer
        self.log = logger or _log
        self.capture_payloads = capture_payloads
        self.tracing_capture_payloads = tracing_capture_payloads
        self.max_payload_chars = max_payload_chars

    # -- 请求作用域 -------------------------------------------------------------

    def bind_request(
        self,
        *,
        session_id: str = "",
        conversation_id: str = "",
        trace_id: str | None = None,
    ) -> TraceContext:
        """绑定本次请求的追踪身份，返回新上下文。"""
        context = bind_trace(
            session_id=session_id, conversation_id=conversation_id, trace_id=trace_id
        )
        self._log("info", "request_start", session_id=session_id)
        return context

    @contextmanager
    def request_span(
        self,
        *,
        message: str = "",
        mode: str = "",
        model: str = "",
        emit_metric: bool = True,
    ) -> Iterator[Span]:
        """一次用户请求的根 Span；退出时记录耗时与成功状态。

        ``emit_metric=False`` 供子代理使用：它仍是一个 chain Span（追踪里可见），
        但不是一次用户请求，因此不得计入请求级耗时直方图。
        """
        handle = self._start_run(
            name="finharness.turn" if emit_metric else "finharness.subagent",
            run_type="chain",
            inputs=self._payload({"message": message, "mode": mode}),
        )
        span = Span(self, handle)
        span.set_attribute("mode", mode)
        if model:
            span.set_attribute("model", model)
        try:
            yield span
        except BaseException as exc:
            span.set_attribute("status", "error")
            self._finish_span(span, error=classify_error(exc))
            self.record_error(error_type=classify_error(exc), detail=str(exc))
            raise
        else:
            self._finish_span(span, error=span._error)
        finally:
            if emit_metric:
                self._emit(
                    "request_finished",
                    span=span,
                    status=span.attributes.get("status", "ok"),
                    model=span.attributes.get("model", ""),
                )

    # -- LLM --------------------------------------------------------------------

    @asynccontextmanager
    async def llm_span(
        self,
        *,
        model: str,
        call_type: str = "main",
        messages: Any = None,
        tool_count: int = 0,
    ):
        """一次 LLM 调用的子 Span（含压缩摘要与子 Agent）。"""
        inputs: dict[str, Any] = {"model": model, "call_type": call_type, "tool_count": tool_count}
        if self.tracing_capture_payloads and messages is not None:
            inputs["messages"] = self._payload(messages)
        handle = self._start_run(name=f"llm.{call_type}", run_type="llm", inputs=inputs)
        span = Span(self, handle)
        span.set_attribute("model", model)
        span.set_attribute("call_type", call_type)
        try:
            yield span
        except BaseException as exc:
            span.set_attribute("status", "error")
            self._finish_span(span, error=classify_error(exc))
            raise
        else:
            self._finish_span(span, error=span._error)
        finally:
            self._emit("llm_finished", span=span, model=model, call_type=call_type)

    # -- 工具 -------------------------------------------------------------------

    @asynccontextmanager
    async def tool_span(self, *, name: str, args: Any = None, call_id: str = ""):
        """一次工具调用的子 Span。"""
        inputs: dict[str, Any] = {"tool": name, "call_id": call_id}
        if self.capture_payloads and args is not None:
            inputs["args"] = self._payload(args)
        handle = self._start_run(name=f"tool.{name}", run_type="tool", inputs=inputs)
        span = Span(self, handle)
        span.set_attribute("tool", name)
        span.set_attribute("call_id", call_id)
        try:
            yield span
        except BaseException as exc:
            span.set_attribute("status", "error")
            self._finish_span(span, error=classify_error(exc))
            raise
        else:
            self._finish_span(span)

    def tool_finished(self, *, tool: str, status: str, span: Span | None = None) -> None:
        """工具结束后上报调用量与耗时；真正的失败才计入错误总数。

        ``denied``/``blocked``/``loop_guard`` 是治理拦截，不是工具脆弱——把它们
        计成错误会让"哪个工具最脆弱"失真。
        """
        duration_s = span.duration_s() if span is not None else 0.0
        self._emit("tool_finished", tool=tool, status=status, duration_s=duration_s)
        if status not in _GOVERNED_STATUSES:
            self.record_error(error_type="tool_failure", detail=f"{tool}:{status}")

    # -- 错误 -------------------------------------------------------------------

    def record_error(self, *, error_type: str, detail: str = "") -> None:
        """记录一次失败；同时进日志与 ``agent_request_errors_total``。"""
        self._log("warning", "request_error", error_type=error_type, detail=detail)
        if self.metrics is not None:
            try:
                self.metrics.error(error_type=error_type)
            except Exception:  # noqa: BLE001 - 观测尽力而为
                _guard("metrics.error")

    def record_governance_event(self, *, kind: str) -> None:
        """记录一次治理事件（拒绝、配额、审计失败、确认超时），进日志与指标。

        这些事件此前只写日志，无法聚合告警——而"拒绝率突然上升"本身就是需要
        被看见的信号（可能是攻击，也可能是配置错误）。
        """
        self._log("warning", "governance_event", kind=kind)
        if self.metrics is not None:
            try:
                self.metrics.governance_event(kind=kind)
            except Exception:  # noqa: BLE001 - 观测尽力而为
                _guard("metrics.governance_event")

    def log_tool_call(self, *, tool: str, args: Any, turn: int = 0) -> None:
        """工具调用的入参摘要（脱敏后）——排查时最需要的一段。"""
        from finharness.observability.redact import summarize_args

        detail = summarize_args(args) if isinstance(args, dict) else str(args)
        self._log("info", "tool_call", tool=tool, turn=turn, args=detail)

    # -- 内部 -------------------------------------------------------------------

    def _start_run(self, *, name: str, run_type: str, inputs: dict) -> Any | None:
        if self.tracer is None:
            return None
        try:
            parent = _RUN_STACK.get()
            handle = self.tracer.start_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                parent=parent[-1] if parent else None,
            )
            if handle is not None:
                _RUN_STACK.set((*parent, handle))
            return handle
        except Exception:  # noqa: BLE001 - 追踪尽力而为
            _guard("tracer.start_run")
            return None

    def _finish_span(self, span: Span, *, error: str | None = None) -> None:
        if span._ended:
            return
        span._ended = True
        handle = span._handle
        if handle is None or self.tracer is None:
            return
        try:
            stack = _RUN_STACK.get()
            if stack and stack[-1] is handle:
                _RUN_STACK.set(stack[:-1])
            # LLM Span 的用量作为 run 的 outputs，使 LangSmith 能呈现 token 与成本。
            outputs = span.attributes.get("outputs")
            if outputs is None and span._usage is not None:
                usage = span._usage
                outputs = {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                    "cache_hit_tokens": getattr(usage, "cache_hit_tokens", 0),
                    "cache_miss_tokens": getattr(usage, "cache_miss_tokens", 0),
                }
            self.tracer.finish_run(
                handle, attributes=span.attributes, outputs=outputs, error=error
            )
        except Exception:  # noqa: BLE001 - 追踪尽力而为
            _guard("tracer.finish_run")

    def _emit(self, kind: str, **payload: Any) -> None:
        """把 Span 结果送到指标后端；任何失败都只降级。"""
        if self.metrics is None:
            return
        try:
            if kind == "request_finished":
                span: Span = payload["span"]
                self.metrics.request_finished(
                    status=payload["status"],
                    duration_s=span.duration_s(),
                    model=payload.get("model", ""),
                )
                run_finished = getattr(self.metrics, "run_finished", None)
                if callable(run_finished):
                    attrs = span.attributes
                    run_finished(
                        status=payload["status"] or "unknown",
                        reason=str(attrs.get("reason") or "unknown"),
                        rounds=attrs.get("rounds"),
                    )
            elif kind == "llm_finished":
                span: Span = payload["span"]
                usage = span._usage
                self.metrics.llm_finished(
                    model=payload["model"],
                    call_type=payload["call_type"],
                    duration_s=span.duration_s(),
                    first_token_s=span.first_token_s(),
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                    cache_hit_tokens=getattr(usage, "cache_hit_tokens", 0) or 0,
                    cache_miss_tokens=getattr(usage, "cache_miss_tokens", 0) or 0,
                )
            elif kind == "tool_finished":
                self.metrics.tool_finished(
                    tool=payload["tool"],
                    status=payload["status"],
                    duration_s=payload["duration_s"],
                )
        except Exception:  # noqa: BLE001 - 观测尽力而为
            _guard(f"metrics.{kind}")

    def _payload(self, value: Any) -> Any:
        """按需截断并脱敏要记录/上报的完整内容。"""
        from finharness.observability.redact import redact

        try:
            cleaned = redact(value)
        except Exception:  # noqa: BLE001 - 脱敏失败则整体丢弃
            _guard("redact")
            return "<unavailable>"
        if isinstance(cleaned, str) and len(cleaned) > self.max_payload_chars:
            return cleaned[: self.max_payload_chars] + "…"
        return cleaned

    def _log(self, level: str, message: str, **fields: Any) -> None:
        try:
            getattr(self.log, level)(message, extra=fields if fields else None)
        except Exception:  # noqa: BLE001 - 日志尽力而为
            _guard("log")


def build_observer(settings: Any) -> Observer | NullObserver:
    """按配置装配观测门面；后端缺席或未启用时静默降级。

    ``metrics`` 与 ``tracing`` 都要求可选依赖已安装且对应开关为 true；
    任一不满足即跳过该后端，而日志（纯标准库）始终可用。
    """

    section = getattr(settings, "observability", None)
    if section is None:
        return NullObserver()

    metrics = None
    tracer = None

    try:
        if section.metrics.enabled:
            from finharness.observability.metrics import MetricsRecorder

            metrics = MetricsRecorder()
    except Exception:  # noqa: BLE001 - 缺库/构建失败即降级
        _guard("metrics.build")
        metrics = None

    try:
        if section.tracing.enabled:
            from finharness.observability.tracing import LangSmithTracer

            tracer = LangSmithTracer(
                project=section.tracing.project,
                env_key=section.tracing.env_key,
            )
            if not tracer.available:
                tracer = None
    except Exception:  # noqa: BLE001 - 缺库/构建失败即降级
        _guard("tracer.build")
        tracer = None

    if metrics is None and tracer is None:
        return NullObserver()

    logging_section = section.logging
    return Observer(
        metrics=metrics,
        tracer=tracer,
        capture_payloads=logging_section.capture_payloads,
        tracing_capture_payloads=section.tracing.capture_payloads,
        max_payload_chars=logging_section.max_payload_chars,
    )
