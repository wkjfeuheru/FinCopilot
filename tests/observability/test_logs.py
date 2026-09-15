"""结构化 JSON 日志：字段、trace_id 注入与 payload 捕获开关（docs 03.14.1）。"""

import json
import logging

from finharness.observability.context import bind_trace
from finharness.observability.logs import JsonFormatter, TraceContextFilter, setup_logging


def _render(record: logging.LogRecord) -> dict:
    TraceContextFilter().filter(record)
    return json.loads(JsonFormatter().format(record))


def _record(message: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="finharness.test", level=logging.INFO, pathname=__file__,
        lineno=1, msg=message, args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_record_is_single_line_json_with_core_fields() -> None:
    payload = _render(_record("request_start", mode="default"))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "finharness.test"
    assert payload["msg"] == "request_start"
    assert payload["mode"] == "default"
    assert "ts" in payload


def test_trace_id_is_injected_from_context() -> None:
    bind_trace(session_id="s_x", conversation_id="c_y")
    payload = _render(_record("tool_call", tool="get_quote"))

    assert payload["trace_id"].startswith("tr_")
    assert payload["session_id"] == "s_x"
    assert payload["conversation_id"] == "c_y"


def test_extra_fields_are_redacted_before_serialization() -> None:
    payload = _render(_record("tool_call", api_key="placeholder"))

    assert payload["api_key"] == "<redacted>"


def test_setup_logging_reconfigures_on_every_call(tmp_path) -> None:
    """重复调用必须按最新设置重建，否则模块级 create_app 会抢先把文件路径锁死。"""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    setup_logging(level="INFO", json_format=True, path=first)
    bind_trace(session_id="s_1")
    logging.getLogger("finharness.test").info("one")

    setup_logging(level="INFO", json_format=True, path=second)
    logging.getLogger("finharness.test").info("two")

    handlers = logging.getLogger("finharness").handlers
    assert sum(1 for handler in handlers if handler.__class__ is logging.FileHandler) == 1
    assert json.loads(second.read_text(encoding="utf-8").splitlines()[-1])["msg"] == "two"


def test_setup_logging_writes_json_lines_to_file(tmp_path) -> None:
    log_file = tmp_path / "app.jsonl"
    setup_logging(level="INFO", json_format=True, path=log_file)

    bind_trace(session_id="s_file")
    logging.getLogger("finharness.test").info("hello", extra={"answer": 42})

    lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if line]
    assert lines, "日志文件应至少有一行"
    record = json.loads(lines[-1])
    assert record["msg"] == "hello"
    assert record["answer"] == 42
    assert record["trace_id"].startswith("tr_")
