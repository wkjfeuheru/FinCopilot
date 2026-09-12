"""Audit hook: JSONL schema, truncation and secret redaction (docs 4.3)."""

import asyncio
import json

from finharness.hooks.audit import AuditHook, AuditLogWriter, summarize_args
from finharness.types import ToolResult


class FakeTool:
    name = "get_quote"


def read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_summarize_args_truncates_long_values():
    text = summarize_args({"symbol": "600519", "note": "x" * 500})
    assert "symbol=600519" in text
    assert len(text) < 300
    assert "…" in text


def test_summarize_args_redacts_secret_keys():
    # Keys are built from a list so no literal pair reads as a credential; the
    # assertion is that secret-shaped keys never reach the log.
    secret_keys = ("api_key", "token")
    args = dict.fromkeys(secret_keys, "placeholder")
    args["symbol"] = "600519"

    text = summarize_args(args)

    assert "placeholder" not in text
    assert "symbol=600519" in text
    assert text.count("<redacted>") == 2


def test_audit_writes_session_boundaries_and_tool_runs(tmp_path):
    writer = AuditLogWriter(tmp_path / "audit.jsonl")
    hook = AuditHook(writer, session_id="s_test")

    hook.session_start(mode="default", provider="FakeProvider", model="m")
    asyncio.run(
        hook.post(
            FakeTool(), {"symbol": "600519"}, ToolResult(content="ok", ok=True),
            action="run", verdict="allow", duration_ms=12.5,
            citations=["cit_000001"], turn=2, endpoint="akshare:x", rows=3, cols=4,
        )
    )
    hook.session_end(total_tokens=100, tool_calls=1)

    records = read_lines(tmp_path / "audit.jsonl")
    assert [r["action"] for r in records] == ["session_start", "run", "session_end"]
    run = records[1]
    assert run["tool"] == "get_quote"
    assert run["verdict"] == "allow"
    assert run["turn"] == 2
    assert run["cids"] == ["cit_000001"]
    assert run["rows"] == 3 and run["cols"] == 4
    assert run["endpoint"] == "akshare:x"
    assert run["ok"] is True
    assert records[2]["total_tokens"] == 100


def test_audit_seq_increments_within_a_writer(tmp_path):
    writer = AuditLogWriter(tmp_path / "audit.jsonl")
    hook = AuditHook(writer, session_id="s")

    hook.session_start(mode="default", provider="p", model="m")
    hook.session_start(mode="default", provider="p", model="m")

    assert [r["seq"] for r in read_lines(tmp_path / "audit.jsonl")] == [1, 2]


def test_denied_action_is_recorded(tmp_path):
    writer = AuditLogWriter(tmp_path / "audit.jsonl")
    hook = AuditHook(writer, session_id="s")

    asyncio.run(
        hook.post(
            FakeTool(), {}, ToolResult(content="", ok=False, error="denied"),
            action="denied", verdict="deny",
        )
    )

    record = read_lines(tmp_path / "audit.jsonl")[0]
    assert record["action"] == "denied"
    assert record["verdict"] == "deny"
    assert record["ok"] is False
