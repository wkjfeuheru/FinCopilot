"""运行轨迹持久化：把每次 agent 运行的完整 trace 落入 SQLite。

监控平台的存储半边（docs 03.14）：``trace_runs`` 一行一次运行（含输入、
终态、停止/交付原因与全部计数），``trace_rounds`` 一行一个轮次（模型
怎么想、调了什么、返回什么——即 ``AgentTurnOutcome.trace`` 的落盘形态），
``trace_events`` 一行一个非 ``text_delta`` 引擎事件（tool_status、
plan_progress 跑偏、loop_guard 重复调用、交互请求……）。

查询面即 7 项指标：任务完成率、工具调用次数、平均执行步数、工具失败率、
重复调用率、安全拦截次数、超时率——公式见 ``metrics_summary``。

写入纪律：所有公开方法**绝不抛出**。trace 落库是旁路观测，任何失败只留
一条日志；一条轨迹写不进去不该影响正在服务的对话。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from finharness.observability.redact import redact, redact_text, summarize_args
from finharness.utils.clock import utc_now_iso
from finharness.utils.sqlite import SqliteStore

log = logging.getLogger(__name__)

# 捕获文本（thought/answer/preview）不落全文时的截断长度。
TRUNCATE_CHARS = 2000

SCHEMA_TRACE_RUNS = """
CREATE TABLE IF NOT EXISTS trace_runs (
    run_id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'server',
    user_id TEXT NOT NULL DEFAULT '',
    session_id TEXT,
    conversation_id TEXT,
    eval_case_id TEXT,
    input TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    reason TEXT,
    succeeded INTEGER,
    rounds INTEGER,
    tool_calls INTEGER,
    retry_count INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_hit_tokens INTEGER,
    per_agent_json TEXT,
    citations_json TEXT,
    plan_json TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER
)
"""

SCHEMA_TRACE_ROUNDS = """
CREATE TABLE IF NOT EXISTS trace_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    thought TEXT NOT NULL DEFAULT '',
    actions_json TEXT NOT NULL DEFAULT '[]',
    observations_json TEXT NOT NULL DEFAULT '[]',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    llm_first_ms INTEGER NOT NULL DEFAULT 0,
    llm_ms INTEGER NOT NULL DEFAULT 0,
    answer TEXT NOT NULL DEFAULT ''
)
"""

SCHEMA_TRACE_EVENTS = """
CREATE TABLE IF NOT EXISTS trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    turn INTEGER,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
)
"""

INDEX_TRACE_RUNS = """
CREATE INDEX IF NOT EXISTS idx_trace_runs_started
ON trace_runs(started_at DESC)
"""
INDEX_TRACE_RUNS_CONV = """
CREATE INDEX IF NOT EXISTS idx_trace_runs_conversation
ON trace_runs(conversation_id)
"""
INDEX_TRACE_EVENTS_RUN = """
CREATE INDEX IF NOT EXISTS idx_trace_events_run
ON trace_events(run_id, seq)
"""
INDEX_TRACE_ROUNDS_RUN = """
CREATE INDEX IF NOT EXISTS idx_trace_rounds_run
ON trace_rounds(run_id, turn)
"""

# 安全拦截：denied（权限门拒绝）与 blocked（deny 规则命中）。
_BLOCK_VERDICTS = ("denied", "blocked")


class TraceStore(SqliteStore):
    """SQLite trace 库；同一线程模型与 ``MemoryStore`` 一致（短连接 + WAL）。"""

    def __init__(self, db_path: str | Path, *, capture_payloads: bool = True) -> None:
        super().__init__(db_path, check_same_thread=False)
        self.capture_payloads = capture_payloads
        self._seq: dict[str, int] = {}
        # 每个 run 的内存序号；调用方可能在工作线程写 trace，故 get/set/pop 需互斥。
        self._seq_lock = threading.Lock()
        self._init_db()

    # ── 基础设施 ──────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(SCHEMA_TRACE_RUNS)
                connection.execute(SCHEMA_TRACE_ROUNDS)
                connection.execute(SCHEMA_TRACE_EVENTS)
                connection.execute(INDEX_TRACE_RUNS)
                connection.execute(INDEX_TRACE_RUNS_CONV)
                connection.execute(INDEX_TRACE_EVENTS_RUN)
                connection.execute(INDEX_TRACE_ROUNDS_RUN)
        except sqlite3.Error:
            log.exception("trace_store_init_failed path=%s", self.db_path)

    def _text(self, value: str | None) -> str:
        """按捕获策略截断并脱敏自由文本。"""
        if not value:
            return ""
        if not self.capture_payloads:
            value = value[:TRUNCATE_CHARS]
        return redact_text(value)

    # ── 写入（全部吞错）──────────────────────────────────────

    def start_run(
        self,
        *,
        run_id: str,
        source: str = "server",
        user_id: str = "",
        session_id: str | None = None,
        conversation_id: str | None = None,
        eval_case_id: str | None = None,
        input: str = "",
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO trace_runs (run_id, source, user_id, session_id,"
                    " conversation_id, eval_case_id, input, started_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        source,
                        user_id,
                        session_id,
                        conversation_id,
                        eval_case_id,
                        self._text(input),
                        utc_now_iso(timespec="milliseconds"),
                    ),
                )
        except sqlite3.Error:
            log.exception("trace_start_run_failed run_id=%s", run_id)

    def record_event(self, run_id: str, kind: str, payload: dict[str, Any], *, turn: int | None = None) -> None:
        try:
            with self._seq_lock:
                seq = self._seq.get(run_id, 0) + 1
                self._seq[run_id] = seq
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO trace_events (run_id, seq, turn, kind, payload_json)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        run_id,
                        seq,
                        turn if turn is not None else payload.get("turn"),
                        kind,
                        json.dumps(redact(payload), ensure_ascii=False, default=str),
                    ),
                )
        except sqlite3.Error:
            log.exception("trace_record_event_failed run_id=%s kind=%s", run_id, kind)

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        answer: str = "",
        reason: str | None = None,
        succeeded: bool | None = None,
        rounds: int | None = None,
        tool_calls: int | None = None,
        retry_count: int | None = None,
        usage: dict[str, int] | None = None,
        per_agent: dict[str, Any] | None = None,
        citations: list[str] | None = None,
        plan: dict[str, Any] | None = None,
        trace_rounds: list[Any] | None = None,
    ) -> None:
        """写终态并落整份轮次轨迹（``RoundTrace`` 列表，来自 AgentTurnOutcome）。"""
        try:
            usage = usage or {}
            with self._connect() as connection:
                connection.execute(
                    "UPDATE trace_runs SET status=?, answer=?, reason=?, succeeded=?,"
                    " rounds=?, tool_calls=?, retry_count=?, input_tokens=?,"
                    " output_tokens=?, cache_hit_tokens=?, per_agent_json=?,"
                    " citations_json=?, plan_json=?, finished_at=?, duration_ms=?"
                    " WHERE run_id=?",
                    (
                        status,
                        self._text(answer),
                        reason,
                        None if succeeded is None else int(succeeded),
                        rounds,
                        tool_calls,
                        retry_count,
                        usage.get("input_tokens"),
                        usage.get("output_tokens"),
                        usage.get("cache_hit_tokens"),
                        json.dumps(per_agent, ensure_ascii=False, default=str) if per_agent else None,
                        json.dumps(citations, ensure_ascii=False) if citations else None,
                        json.dumps(plan, ensure_ascii=False, default=str) if plan else None,
                        utc_now_iso(timespec="milliseconds"),
                        None,
                        run_id,
                    ),
                )
            if trace_rounds:
                self._write_rounds(run_id, trace_rounds)
            # 终态后清理内存序号。
            with self._seq_lock:
                self._seq.pop(run_id, None)
        except sqlite3.Error:
            log.exception("trace_finish_run_failed run_id=%s", run_id)

    def _write_rounds(self, run_id: str, trace_rounds: list[Any]) -> None:
        rows = []
        for item in trace_rounds:
            actions = [
                {"call_id": a.call_id, "name": a.name, "args": summarize_args(a.args)}
                for a in getattr(item, "actions", []) or []
            ]
            observations = [
                {
                    "call_id": o.call_id,
                    "name": o.name,
                    "ok": bool(o.ok),
                    "error": o.error,
                    "preview": self._text(o.preview),
                    "duration_ms": o.duration_ms,
                }
                for o in getattr(item, "observations", []) or []
            ]
            rows.append(
                (
                    run_id,
                    item.turn,
                    self._text(getattr(item, "thought", "") or ""),
                    json.dumps(actions, ensure_ascii=False),
                    json.dumps(observations, ensure_ascii=False),
                    getattr(item, "input_tokens", 0) or 0,
                    getattr(item, "output_tokens", 0) or 0,
                    getattr(item, "llm_first_ms", 0) or 0,
                    getattr(item, "llm_ms", 0) or 0,
                    self._text(getattr(item, "answer", "") or ""),
                )
            )
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO trace_rounds (run_id, turn, thought, actions_json,"
                " observations_json, input_tokens, output_tokens, llm_first_ms,"
                " llm_ms, answer) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def cleanup(self, retention_days: int) -> int:
        """按保留期清理过期运行；返回删除的行数（0 或失败时为 0）。"""
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat(
            timespec="milliseconds"
        )
        removed = 0
        try:
            with self._connect() as connection:
                old_ids = [
                    row["run_id"]
                    for row in connection.execute(
                        "SELECT run_id FROM trace_runs WHERE started_at < ?", (cutoff,)
                    )
                ]
                if not old_ids:
                    return 0
                marks = ",".join("?" * len(old_ids))
                connection.execute(f"DELETE FROM trace_events WHERE run_id IN ({marks})", old_ids)
                connection.execute(f"DELETE FROM trace_rounds WHERE run_id IN ({marks})", old_ids)
                connection.execute(f"DELETE FROM trace_runs WHERE run_id IN ({marks})", old_ids)
                removed = len(old_ids)
        except sqlite3.Error:
            log.exception("trace_cleanup_failed")
        return removed

    # ── 查询 ─────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str | None) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def list_runs(
        self,
        *,
        source: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where, params = self._run_filters(
            source=source, status=status, user_id=user_id,
            conversation_id=conversation_id, since=since, until=until,
        )
        sql = "SELECT * FROM trace_runs" + where + " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params = [*params, int(limit), int(offset)]
        try:
            with self._connect() as connection:
                rows = connection.execute(sql, params).fetchall()
                return [self._run_row(dict(row)) for row in rows]
        except sqlite3.Error:
            log.exception("trace_list_runs_failed")
            return []

    def count_runs(self, **filters: Any) -> int:
        where, params = self._run_filters(**filters)
        sql = "SELECT COUNT(*) AS n FROM trace_runs" + where
        try:
            with self._connect() as connection:
                return int(connection.execute(sql, params).fetchone()["n"])
        except sqlite3.Error:
            log.exception("trace_count_runs_failed")
            return 0

    @staticmethod
    def _run_filters(
        *, source: str | None = None, status: str | None = None,
        user_id: str | None = None, conversation_id: str | None = None,
        since: str | None = None, until: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if since:
            clauses.append("started_at >= ?")
            params.append(since)
        if until:
            clauses.append("started_at < ?")
            params.append(until)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    @staticmethod
    def _run_row(row: dict[str, Any]) -> dict[str, Any]:
        for key in ("per_agent_json", "citations_json", "plan_json"):
            short = key.removesuffix("_json")
            row[short] = TraceStore._parse_json(row.pop(key))
        return row

    def run_detail(self, run_id: str) -> dict[str, Any] | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM trace_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is None:
                    return None
                detail = self._run_row(dict(row))
                detail["rounds_trace"] = [
                    self._round_row(dict(r))
                    for r in connection.execute(
                        "SELECT * FROM trace_rounds WHERE run_id = ? ORDER BY turn",
                        (run_id,),
                    )
                ]
                detail["events"] = [
                    {
                        "seq": e["seq"],
                        "turn": e["turn"],
                        "kind": e["kind"],
                        "payload": TraceStore._parse_json(e["payload_json"]),
                    }
                    for e in connection.execute(
                        "SELECT * FROM trace_events WHERE run_id = ? ORDER BY seq",
                        (run_id,),
                    )
                ]
                return detail
        except sqlite3.Error:
            log.exception("trace_run_detail_failed run_id=%s", run_id)
            return None

    @staticmethod
    def _round_row(row: dict[str, Any]) -> dict[str, Any]:
        row["actions"] = TraceStore._parse_json(row.pop("actions_json")) or []
        row["observations"] = TraceStore._parse_json(row.pop("observations_json")) or []
        return row

    # ── 指标聚合（7 项 + 跑偏分布）────────────────────────────

    def metrics_summary(
        self,
        *,
        source: str | None = None,
        user_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """七项运行指标 + 停止原因拆解 + 每工具失败榜 + 首次跑偏轮次分布。"""
        where, params = self._run_filters(source=source, user_id=user_id, since=since, until=until)
        result: dict[str, Any] = {
            "generated_at": utc_now_iso(timespec="milliseconds"),
            "filters": {"source": source, "user_id": user_id, "since": since, "until": until},
        }
        try:
            with self._connect() as connection:
                runs = connection.execute(
                    "SELECT run_id, status, reason, succeeded, rounds FROM trace_runs" + where,
                    params,
                ).fetchall()
                events = self._events_for_runs(
                    connection, [r["run_id"] for r in runs]
                )
        except sqlite3.Error:
            log.exception("trace_metrics_failed")
            return {**result, "error": "metrics_query_failed"}

        total = len(runs)
        done = sum(1 for r in runs if r["status"] == "done")
        stopped = sum(1 for r in runs if r["status"] == "stopped")
        error = sum(1 for r in runs if r["status"] == "error")
        aborted = sum(1 for r in runs if r["status"] == "aborted")
        succeeded = sum(1 for r in runs if r["succeeded"])
        rounds_values = [r["rounds"] for r in runs if r["rounds"] is not None]
        reason_counts: dict[str, int] = {}
        for r in runs:
            key = r["reason"] or r["status"]
            reason_counts[key] = reason_counts.get(key, 0) + 1

        tool_calls = [e for e in events if e["kind"] == "tool_status"]
        started = sum(1 for e in tool_calls if (e["payload"] or {}).get("status") == "started")
        failed = [
            e for e in tool_calls
            if (e["payload"] or {}).get("status") == "failed"
            and (e["payload"] or {}).get("verdict") not in _BLOCK_VERDICTS
        ]
        blocked = [
            e for e in tool_calls
            if (e["payload"] or {}).get("verdict") in _BLOCK_VERDICTS
        ]
        timed_out = [
            e for e in tool_calls
            if (e["payload"] or {}).get("verdict") == "timeout"
            or ((e["payload"] or {}).get("status") == "failed" and "超时" in str((e["payload"] or {}).get("error") or ""))
        ]
        loop_guards = [e for e in events if e["kind"] == "loop_guard"]
        per_tool: dict[str, dict[str, int]] = {}
        for e in failed:
            name = (e["payload"] or {}).get("name") or "unknown"
            entry = per_tool.setdefault(name, {"failed": 0, "blocked": 0, "total": 0})
            entry["failed"] += 1
        for e in blocked:
            name = (e["payload"] or {}).get("name") or "unknown"
            entry = per_tool.setdefault(name, {"failed": 0, "blocked": 0, "total": 0})
            entry["blocked"] += 1
        for e in tool_calls:
            payload = e["payload"] or {}
            # total 只数真正开始执行的调用（started），与全局口径一致；
            # 终态帧（completed/failed）是同一调用的第二个事件，重复计入会
            # 让单工具失败率与全局失败率对不上。
            if payload.get("status") != "started":
                continue
            name = payload.get("name") or "unknown"
            per_tool.setdefault(name, {"failed": 0, "blocked": 0, "total": 0})["total"] += 1

        drift_first_turn: dict[str, int] = {}
        for e in events:
            payload = e["payload"] or {}
            if e["kind"] == "plan_progress":
                drift = payload.get("drift") or (payload.get("plan") or {}).get("drift")
                if drift:
                    key = str(e["turn"] if e["turn"] is not None else payload.get("turn") or "?")
                    drift_first_turn.setdefault(key, 0)
                    drift_first_turn[key] += 1

        def _ratio(numerator: int, denominator: int) -> float | None:
            return round(numerator / denominator, 4) if denominator else None

        result.update(
            {
                "total_runs": total,
                "completion": {
                    "task_completion_rate": _ratio(succeeded, total),
                    "status_counts": {"done": done, "stopped": stopped, "error": error, "aborted": aborted},
                    "reason_counts": reason_counts,
                },
                "tool_calls_total": started,
                "avg_rounds": round(sum(rounds_values) / len(rounds_values), 2) if rounds_values else None,
                "tool_failure_rate": _ratio(len(failed), started),
                "repeat_call_rate": _ratio(len(loop_guards), started),
                "loop_guard_events": len(loop_guards),
                "safety_blocks": len(blocked),
                "timeout_rate": _ratio(len(timed_out), started),
                "per_tool": {k: {**v, "failure_rate": _ratio(v["failed"], v["total"])} for k, v in per_tool.items()},
                "drift_first_turn": drift_first_turn,
            }
        )
        return result

    @staticmethod
    def _events_for_runs(connection: sqlite3.Connection, run_ids: list[str]) -> list[dict[str, Any]]:
        if not run_ids:
            return []
        events: list[dict[str, Any]] = []
        chunk = 500
        for i in range(0, len(run_ids), chunk):
            part = run_ids[i : i + chunk]
            marks = ",".join("?" * len(part))
            events.extend(
                {"kind": r["kind"], "turn": r["turn"], "payload": TraceStore._parse_json(r["payload_json"])}
                for r in connection.execute(
                    f"SELECT kind, turn, payload_json FROM trace_events WHERE run_id IN ({marks})",
                    part,
                )
            )
        return events


__all__ = ["TraceStore", "TRUNCATE_CHARS"]
