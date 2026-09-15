"""可观测性：结构化日志、Prometheus 指标与追踪（docs 03.14）。

调用方只需依赖 ``Observer`` 门面；未安装可选依赖或未启用开关时，
``build_observer`` 返回 ``NullObserver``，引擎与测试无需任何改动。
"""

from __future__ import annotations

from finharness.observability.context import (
    TraceContext,
    bind_trace,
    current_trace,
    new_trace_id,
    update_turn,
)
from finharness.observability.logs import get_logger, setup_logging
from finharness.observability.observer import (
    NullObserver,
    Observer,
    Span,
    build_observer,
    classify_error,
    current_trace_id,
)
from finharness.observability.redact import redact, redact_text, summarize_args

__all__ = [
    "NullObserver",
    "Observer",
    "Span",
    "TraceContext",
    "bind_trace",
    "build_observer",
    "classify_error",
    "current_trace",
    "current_trace_id",
    "get_logger",
    "new_trace_id",
    "redact",
    "redact_text",
    "setup_logging",
    "summarize_args",
    "update_turn",
]
