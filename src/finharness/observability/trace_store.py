"""运行轨迹持久化：把每次 agent 运行的完整 trace 落入 SQLite。

监控平台的存储半边（docs 03.14）：``trace_runs`` 一行一次运行（含输入、
终态、停止/交付原因与全部计数），``trace_rounds`` 一行一个轮次（模型
怎么想、调了什么、返回什么——即 ``AgentTurnOutcome.trace`` 的落盘形态），
``trace_states`` 一行一次 FSM 状态转换（每步 agent 的 state，按 revision
单调），``trace_events`` 一行一个非 ``text_delta`` 引擎事件（tool_status、
plan_progress 跑偏、loop_guard 重复调用、交互请求……）。

``trace_rounds`` 与 ``trace_states`` 都可在运行进行中实时落库
（``record_round`` / ``record_event`` 的 state 分支），因此进程崩溃不会
丢掉已发生的轮次与状态；``finish_run`` 的批量写与之共用 upsert，幂等共存。

查询面即 7 项指标 + 步数：任务完成率、工具调用次数、平均执行步数、工具
失败率、重复调用率、安全拦截次数、超时率——公式见 ``metrics_summary``。

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
    answer TEXT NOT NULL DEFAULT '',
    phase TEXT,
    revision INTEGER
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

# 每步 FSM state 的一等记录（docs 03.14）：一行一次合法状态转换。
# ``run_id`` 是 trace 库的运行 id（与 ``trace_runs`` 关联），``agent_run_id``
# 是该事件的 FSM 运行身份（resume 时不变），二者不同——前者按 HTTP 请求
# 分段，后者把各段串回同一次 agent 运行。
SCHEMA_TRACE_STATES = """
CREATE TABLE IF NOT EXISTS trace_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_run_id TEXT NOT NULL DEFAULT '',
    revision INTEGER NOT NULL,
    turn INTEGER,
    phase TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
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
# 唯一键让实时写、finish_run 批量写与 eval 重放三条路径幂等共存
# （``INSERT … ON CONFLICT`` 的冲突目标），同时服务于按 run 的查询。
INDEX_TRACE_ROUNDS_RUN = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_trace_rounds_run_turn
ON trace_rounds(run_id, turn)
"""

INDEX_TRACE_STATES_RUN = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_trace_states_run_revision
ON trace_states(run_id, revision)
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
                connection.execute(SCHEMA_TRACE_STATES)
                connection.execute(INDEX_TRACE_RUNS)
                connection.execute(INDEX_TRACE_RUNS_CONV)
                connection.execute(INDEX_TRACE_EVENTS_RUN)
                self._migrate(connection)
                # 唯一索引建在迁移之后：旧库可能残留重复的 ``(run_id, turn)``
                # 行，必须先收敛再建唯一键，否则 CREATE UNIQUE INDEX 直接失败。
                connection.execute(INDEX_TRACE_ROUNDS_RUN)
                connection.execute(INDEX_TRACE_STATES_RUN)
        except sqlite3.Error:
            log.exception("trace_store_init_failed path=%s", self.db_path)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """幂等迁移旧库：补 trace_rounds 的 phase/revision 列并去重。

        与 ``MemoryStore._migrate`` 同一模式（``PRAGMA table_info`` 探测后再
        ``ALTER TABLE ADD COLUMN``），可重入、不动既有数据。
        """
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(trace_rounds)")
        }
        if "phase" not in columns:
            connection.execute("ALTER TABLE trace_rounds ADD COLUMN phase TEXT")
        if "revision" not in columns:
            connection.execute("ALTER TABLE trace_rounds ADD COLUMN revision INTEGER")
        # 若唯一索引尚未建立，说明是旧库（或首次升级）：历史写路径
        # （eval 重跑等）可能在 ``(run_id, turn)`` 上留下重复行，此时只保留
        # 每个键的最新一行（id 最大），使实时/finish 两条 upsert 路径可安全
        # 共存。索引已存在时跳过整表去重，避免每次启动都全表扫描。
        has_unique = any(
            row["name"] == "idx_trace_rounds_run_turn"
            for row in connection.execute("PRAGMA index_list(trace_rounds)")
        )
        if not has_unique:
            connection.execute(
                "DELETE FROM trace_rounds WHERE id NOT IN"
                " (SELECT MAX(id) FROM trace_rounds GROUP BY run_id, turn)"
            )

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
                # ``state`` 事件顺带落一行 trace_states，把每步 FSM 状态提升为
                # 一等产物。服务端实时路径与 eval 批量重放都经由本方法，故
                # 无需改动任何 producer 即可同时覆盖两条写入路径。
                if kind == "state":
                    self._insert_state(connection, run_id, payload)
        except sqlite3.Error:
            log.exception("trace_record_event_failed run_id=%s kind=%s", run_id, kind)

    def _insert_state(
        self, connection: sqlite3.Connection, run_id: str, payload: dict[str, Any]
    ) -> None:
        """把一次公共状态视图的转换追加进 ``trace_states``；``(run_id, revision)`` 幂等。"""
        revision = payload.get("revision")
        if not isinstance(revision, int):
            # revision 是步进身份的锚点；缺失时放弃落库而非写入不可关联的行。
            log.warning("trace_state_missing_revision run_id=%s", run_id)
            return
        agent_run_id = payload.get("run_id")
        phase = payload.get("phase")
        turn = payload.get("turn")
        created_at = (
            payload.get("updated_at")
            or payload.get("created_at")
            or utc_now_iso(timespec="milliseconds")
        )
        connection.execute(
            "INSERT INTO trace_states (run_id, agent_run_id, revision, turn, phase,"
            " created_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(run_id, revision) DO UPDATE SET"
            " agent_run_id=excluded.agent_run_id, turn=excluded.turn,"
            " phase=excluded.phase, created_at=excluded.created_at,"
            " payload_json=excluded.payload_json",
            (
                run_id,
                str(agent_run_id or ""),
                revision,
                turn if isinstance(turn, int) else None,
                str(phase or ""),
                str(created_at),
                json.dumps(redact(payload), ensure_ascii=False, default=str),
            ),
        )

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

    # 轮次写入统一走 upsert：实时（``record_round``）、run 结束批量
    # （``_write_rounds``）与 eval 重放三条路径因此幂等共存。批量路径不带
    # phase/revision（None），故用 COALESCE 保留实时路径已写的值，避免被
    # 覆盖成 NULL。
    _ROUND_UPSERT = (
        "INSERT INTO trace_rounds (run_id, turn, thought, actions_json,"
        " observations_json, input_tokens, output_tokens, llm_first_ms,"
        " llm_ms, answer, phase, revision)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(run_id, turn) DO UPDATE SET"
        " thought=excluded.thought, actions_json=excluded.actions_json,"
        " observations_json=excluded.observations_json,"
        " input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens,"
        " llm_first_ms=excluded.llm_first_ms, llm_ms=excluded.llm_ms,"
        " answer=excluded.answer,"
        " phase=COALESCE(excluded.phase, trace_rounds.phase),"
        " revision=COALESCE(excluded.revision, trace_rounds.revision)"
    )

    def _round_values(
        self,
        run_id: str,
        item: Any,
        *,
        phase: str | None = None,
        revision: int | None = None,
    ) -> tuple[Any, ...]:
        """把一个 ``RoundTrace`` 转成 ``trace_rounds`` 的一行绑定值。"""
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
        return (
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
            phase,
            revision if isinstance(revision, int) else None,
        )

    def record_round(
        self,
        run_id: str,
        round_trace: Any,
        *,
        phase: str | None = None,
        revision: int | None = None,
    ) -> None:
        """实时落一条轮次轨迹（崩溃不丢）；``(run_id, turn)`` 幂等。

        在 run 进行中由 ``OutputSink.record_round`` 调用，与 ``finish_run``
        的批量写共用同一 upsert，因此同一轮重复到达时只保留最后一次。
        """
        try:
            with self._connect() as connection:
                connection.execute(
                    self._ROUND_UPSERT, self._round_values(run_id, round_trace, phase=phase, revision=revision)
                )
        except sqlite3.Error:
            log.exception("trace_record_round_failed run_id=%s", run_id)

    def _write_rounds(self, run_id: str, trace_rounds: list[Any]) -> None:
        rows = [self._round_values(run_id, item) for item in trace_rounds]
        with self._connect() as connection:
            connection.executemany(self._ROUND_UPSERT, rows)

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
                connection.execute(f"DELETE FROM trace_states WHERE run_id IN ({marks})", old_ids)
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
                # 每步 FSM state（按 revision 单调）：阶段时间线、转换明细的
                # 一等来源；相位时长由前端用相邻 ``created_at`` 计算。
                detail["states"] = [
                    TraceStore._state_row(dict(s))
                    for s in connection.execute(
                        "SELECT * FROM trace_states WHERE run_id = ? ORDER BY revision",
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

    @staticmethod
    def _state_row(row: dict[str, Any]) -> dict[str, Any]:
        row["payload"] = TraceStore._parse_json(row.pop("payload_json")) or {}
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
        """七项运行指标 + 停止原因拆解 + 每工具失败榜 + 首次跑偏轮次分布 + 步数。"""
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
                run_ids = [r["run_id"] for r in runs]
                events = self._events_for_runs(connection, run_ids)
                states_total, states_per_run = self._state_counts(connection, run_ids)
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
                # 每步 FSM state 的步数口径：总步数与平均每运行步数。
                "total_states": states_total,
                "avg_steps": round(sum(states_per_run) / len(states_per_run), 2)
                if states_per_run
                else None,
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
    def _state_counts(
        connection: sqlite3.Connection, run_ids: list[str]
    ) -> tuple[int, list[int]]:
        """每运行的状态步数（行数）；返回 (总步数, 每运行步数列表)。

        只统计有状态记录的运行，使 ``avg_steps`` 不会被旧运行（无
        ``trace_states``）拉低——与 ``avg_rounds`` 忽略 NULL 的纪律一致。
        """
        if not run_ids:
            return 0, []
        counts: dict[str, int] = {}
        chunk = 500
        for i in range(0, len(run_ids), chunk):
            part = run_ids[i : i + chunk]
            marks = ",".join("?" * len(part))
            for row in connection.execute(
                f"SELECT run_id, COUNT(*) AS n FROM trace_states"
                f" WHERE run_id IN ({marks}) GROUP BY run_id",
                part,
            ):
                counts[row["run_id"]] = int(row["n"])
        return sum(counts.values()), list(counts.values())

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
