"""结构化 JSON 日志：每条记录都携带 ``trace_id``，便于串联一次对话（docs 03.14）。

只用标准库 ``logging``：项目其余部分没有任何日志设施，因此这里定义唯一的入口
``setup_logging`` 与 ``get_logger``，避免日志平台因裸文本无法做字段级过滤。
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from finharness.observability.context import current_trace
from finharness.observability.redact import redact_field

__all__ = ["JsonFormatter", "TraceContextFilter", "get_logger", "setup_logging"]

# LogRecord 的内建属性；不属于这些的键视为调用方附加的字段。
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "trace_id", "session_id", "conversation_id", "turn"}


class TraceContextFilter(logging.Filter):
    """把当前追踪上下文注入每条记录，使日志无需手工传 id。"""

    def filter(self, record: logging.LogRecord) -> bool:
        trace = current_trace()
        record.trace_id = trace.trace_id if trace else ""
        record.session_id = trace.session_id if trace else ""
        record.conversation_id = trace.conversation_id if trace else ""
        record.turn = trace.turn if trace else 0
        return True


class JsonFormatter(logging.Formatter):
    """把日志记录渲染成单行 JSON；附加字段并入顶层。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", "")
        if trace_id:
            payload["trace_id"] = trace_id
        for name in ("session_id", "conversation_id", "turn"):
            value = getattr(record, name, None)
            if value:
                payload[name] = value
        # 调用方通过 ``extra=`` 附加的字段直接并入，但绝不让凭据进入日志。
        # 日志字段是扁平的具名属性，因此按键名判断而不是只看值。
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = redact_field(key, value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    *,
    level: str = "INFO",
    json_format: bool = True,
    path: str | Path | None = None,
) -> None:
    """配置 ``finharness`` logger：JSON 输出到 stdout，可选再追加一份到文件。

    每次调用都**重新配置**（先移除本函数此前添加的处理器）。这是刻意的：
    ``server/api.py`` 在模块级会先以默认设置建一次 app，之后测试或调用方再用
    真实设置建一次；若这里选择"已有处理器就跳过"，第二次调用就会静默失效，
    日志文件永远不会被创建。本函数是该 logger 处理器的唯一所有者，因此它有权
    按最新设置重建。
    """

    root = logging.getLogger("finharness")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()

    formatter: logging.Formatter = JsonFormatter() if json_format else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(trace_id)s] %(message)s"
    )
    context_filter = TraceContextFilter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(context_filter)
    root.addHandler(stream)

    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)

    root.setLevel(level.upper())
    # 不向 root 冒泡，避免与 uvicorn/第三方 logger 重复输出。
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """获取 ``finharness`` 命名空间下的 logger。"""
    suffix = name[len("finharness.") :] if name.startswith("finharness.") else name
    return logging.getLogger(f"finharness.{suffix}")
