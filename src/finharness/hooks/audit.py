"""Audit hook: append-only JSONL record of every governed tool call (docs 4.3).

Always on and not removable — ``settings.audit`` chooses the path, never the
switch. Each line is flushed immediately so a crash still leaves a complete
record of what ran.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from finharness.hooks.base import BaseHook
from finharness.types import ToolResult

MAX_ARG_VALUE_LEN = 200
REDACTED_KEYS = ("api_key", "token", "secret", "password", "authorization")


class AuditLogWriter:
    """Line-buffered JSONL writer; one line per event."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def write(self, record: dict) -> None:
        self._seq += 1
        payload = {"seq": self._seq, "ts": datetime.now().astimezone().isoformat(timespec="seconds")}
        payload.update(record)
        with open(self.path, "a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def summarize_args(args: dict) -> str:
    """Render an argument snapshot, truncating long values and redacting secrets."""
    parts: list[str] = []
    for key, value in args.items():
        if key.lower() in REDACTED_KEYS:
            parts.append(f"{key}=<redacted>")
            continue
        text = str(value)
        if len(text) > MAX_ARG_VALUE_LEN:
            text = text[:MAX_ARG_VALUE_LEN] + "…"
        parts.append(f"{key}={text}")
    return ",".join(parts)


class AuditHook(BaseHook):
    """Writes session boundaries and one line per tool outcome."""

    def __init__(self, writer: AuditLogWriter, *, session_id: str = "local") -> None:
        self.writer = writer
        self.session_id = session_id

    def session_start(self, *, mode: str, provider: str, model: str) -> None:
        self.writer.write(
            {
                "session_id": self.session_id,
                "action": "session_start",
                "mode": mode,
                "provider": provider,
                "model": model,
            }
        )

    def session_end(self, *, total_tokens: int, tool_calls: int) -> None:
        self.writer.write(
            {
                "session_id": self.session_id,
                "action": "session_end",
                "total_tokens": total_tokens,
                "tool_calls": tool_calls,
            }
        )

    async def post(
        self,
        tool,
        args: dict,
        result: ToolResult,
        *,
        action: str,
        verdict: str,
        duration_ms: float = 0.0,
        citations: list[str] | None = None,
        turn: int = 0,
        endpoint: str = "",
        rows: int = 0,
        cols: int = 0,
    ) -> None:
        self.writer.write(
            {
                "session_id": self.session_id,
                "action": action,
                "turn": turn,
                "tool": getattr(tool, "name", "unknown"),
                "verdict": verdict,
                "args_summary": summarize_args(args),
                "duration_ms": duration_ms,
                "ok": bool(result.ok),
                "cids": list(citations or []),
                "rows": rows,
                "cols": cols,
                "endpoint": endpoint,
            }
        )
