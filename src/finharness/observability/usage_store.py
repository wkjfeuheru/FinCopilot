"""用量账本：每轮对话一行的 token/轮次记录（管理员页数据源）。

与 ``TraceStore`` 的分工：trace 是**质量观测**（可选、含正文、有保留期），
账本是**计费口径**（永远开启、只有元数据、永久累积）。管理员页的
"谁花了多少 token / 聊了多少轮"从这里聚合，不依赖 trace 开关。

写入纪律与 ``TraceStore`` 一致：公开方法**绝不抛出**。落账是旁路观测，
任何失败只留一条日志；一行用量写不进去不该影响正在服务的对话。
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_USAGE_TURNS = """
CREATE TABLE IF NOT EXISTS usage_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'done'
)
"""

INDEX_USAGE_TURNS_USER = """
CREATE INDEX IF NOT EXISTS idx_usage_turns_user ON usage_turns(user_id, ts)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class UsageStore:
    """SQLite 用量账本；短连接 + WAL，与 ``UserStore`` 同一模型。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(SCHEMA_USAGE_TURNS)
            connection.execute(INDEX_USAGE_TURNS_USER)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def ping(self) -> None:
        """就绪检查：账本可连接可查询。"""
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()

    def record_turn(
        self,
        *,
        user_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit_tokens: int = 0,
        duration_ms: int | None = None,
        status: str = "done",
        ts: str | None = None,
    ) -> None:
        """记录一轮对话的用量；任何失败只降级为日志。"""
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO usage_turns (user_id, ts, input_tokens,"
                    " output_tokens, cache_hit_tokens, duration_ms, status)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        ts or _now(),
                        int(input_tokens or 0),
                        int(output_tokens or 0),
                        int(cache_hit_tokens or 0),
                        duration_ms,
                        status,
                    ),
                )
        except sqlite3.Error:
            log.exception("usage_record_failed")

    def totals_by_user(self, *, since: str | None = None) -> list[dict[str, Any]]:
        """按用户聚合（可带时间窗下限）：轮数与三类 token 的总量。"""
        query = (
            "SELECT user_id, COUNT(*) AS turns, SUM(input_tokens) AS input_tokens,"
            " SUM(output_tokens) AS output_tokens,"
            " SUM(cache_hit_tokens) AS cache_hit_tokens"
            " FROM usage_turns"
        )
        params: tuple[str, ...] = ()
        if since is not None:
            query += " WHERE ts >= ?"
            params = (since,)
        query += " GROUP BY user_id ORDER BY turns DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def summary(self, *, since: str | None = None) -> dict[str, Any]:
        """全局汇总：窗口内轮数、活跃用户数、token 总量（汇总卡片数据源）。"""
        query = (
            "SELECT COUNT(*) AS turns, COUNT(DISTINCT user_id) AS active_users,"
            " SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,"
            " SUM(cache_hit_tokens) AS cache_hit_tokens"
            " FROM usage_turns"
        )
        params: tuple[str, ...] = ()
        if since is not None:
            query += " WHERE ts >= ?"
            params = (since,)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return {
            "turns": int(row["turns"] or 0),
            "active_users": int(row["active_users"] or 0),
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "cache_hit_tokens": int(row["cache_hit_tokens"] or 0),
        }
