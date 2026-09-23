"""多租户计算任务的 SQLite 队列。

worker 只通过受控接口租用任务；租户归属、配额与状态转换始终由主服务保存。
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class QueueFullError(RuntimeError):
    """用户的等待队列已满。"""


@dataclass(frozen=True, slots=True)
class ComputeJob:
    job_id: str
    user_id: str
    conversation_id: str
    kind: str
    payload_path: str
    status: str
    attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    result_json: str | None
    error: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS compute_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued', 'leased', 'running', 'succeeded', 'failed', 'cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at REAL,
    result_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compute_jobs_pick ON compute_jobs(status, user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_compute_jobs_user ON compute_jobs(user_id, status, created_at);
CREATE TABLE IF NOT EXISTS compute_queue_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class ComputeJobStore:
    """任务状态机：queued → leased → running → 终态。"""

    def __init__(self, db_path: str | Path, *, max_waiting_per_user: int = 2, max_attempts: int = 2) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_waiting_per_user = max_waiting_per_user
        self.max_attempts = max_attempts
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _row(row: sqlite3.Row) -> ComputeJob:
        return ComputeJob(
            job_id=row["job_id"], user_id=row["user_id"], conversation_id=row["conversation_id"],
            kind=row["kind"], payload_path=row["payload_path"], status=row["status"],
            attempts=int(row["attempts"]), lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"], result_json=row["result_json"], error=row["error"],
        )

    def enqueue(self, *, user_id: str, conversation_id: str, kind: str, payload_path: str) -> ComputeJob:
        now = time.time()
        with self._connect() as connection:
            waiting = connection.execute(
                "SELECT COUNT(*) AS n FROM compute_jobs WHERE user_id = ? AND status = 'queued'", (user_id,)
            ).fetchone()["n"]
            if int(waiting) >= self.max_waiting_per_user:
                raise QueueFullError("该用户的计算任务等待队列已满")
            job_id = f"job_{uuid.uuid4().hex}"
            connection.execute(
                "INSERT INTO compute_jobs (job_id, user_id, conversation_id, kind, payload_path, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, user_id, conversation_id, kind, payload_path, now, now),
            )
            row = connection.execute("SELECT * FROM compute_jobs WHERE job_id = ?", (job_id,)).fetchone()
        assert row is not None
        return self._row(row)

    def lease_next(self, *, worker_id: str, lease_seconds: float) -> ComputeJob | None:
        now = time.time()
        expiry = now + max(float(lease_seconds), 0.0)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired(connection, now)
            previous = connection.execute(
                "SELECT value FROM compute_queue_state WHERE key = 'last_user_id'"
            ).fetchone()
            last_user = previous["value"] if previous else ""
            row = connection.execute(
                "SELECT q.* FROM compute_jobs q "
                "WHERE q.status = 'queued' "
                "AND NOT EXISTS (SELECT 1 FROM compute_jobs active WHERE active.user_id = q.user_id "
                "AND active.status IN ('leased', 'running')) "
                "ORDER BY CASE WHEN q.user_id = ? THEN 1 ELSE 0 END, q.created_at, q.job_id LIMIT 1",
                (last_user,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            cursor = connection.execute(
                "UPDATE compute_jobs SET status = 'leased', attempts = attempts + 1, lease_owner = ?, "
                "lease_expires_at = ?, updated_at = ? WHERE job_id = ? AND status = 'queued'",
                (worker_id, expiry, now, row["job_id"]),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.execute(
                "INSERT INTO compute_queue_state (key, value) VALUES ('last_user_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (row["user_id"],),
            )
            leased = connection.execute("SELECT * FROM compute_jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
            connection.commit()
        assert leased is not None
        return self._row(leased)

    def _recover_expired(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            "UPDATE compute_jobs SET status = 'queued', lease_owner = NULL, lease_expires_at = NULL, updated_at = ? "
            "WHERE status IN ('leased', 'running') AND lease_expires_at <= ? AND attempts < ?",
            (now, now, self.max_attempts),
        )
        connection.execute(
            "UPDATE compute_jobs SET status = 'failed', error = 'worker lease expired', lease_owner = NULL, "
            "lease_expires_at = NULL, updated_at = ? WHERE status IN ('leased', 'running') "
            "AND lease_expires_at <= ? AND attempts >= ?",
            (now, now, self.max_attempts),
        )

    def mark_running(self, job_id: str, *, worker_id: str, lease_seconds: float) -> None:
        self._transition_lease(job_id, worker_id=worker_id, status="running", lease_seconds=lease_seconds)

    def renew(self, job_id: str, *, worker_id: str, lease_seconds: float) -> None:
        self._transition_lease(job_id, worker_id=worker_id, status=None, lease_seconds=lease_seconds)

    def _transition_lease(self, job_id: str, *, worker_id: str, status: str | None, lease_seconds: float) -> None:
        now = time.time()
        assignments = [now + max(float(lease_seconds), 0.0), now, job_id, worker_id, now]
        sql = "UPDATE compute_jobs SET lease_expires_at = ?, updated_at = ?"
        if status is not None:
            sql += ", status = ?"
            assignments.insert(2, status)
        sql += " WHERE job_id = ? AND lease_owner = ? AND status IN ('leased', 'running') AND lease_expires_at > ?"
        with self._connect() as connection:
            if connection.execute(sql, assignments).rowcount != 1:
                raise ValueError("任务未被当前 worker 租用或租约已过期")

    def succeed(self, job_id: str, *, worker_id: str, result_json: str) -> None:
        self._finish(job_id, worker_id=worker_id, status="succeeded", result_json=result_json, error=None)

    def fail(self, job_id: str, *, worker_id: str, error: str) -> None:
        self._finish(job_id, worker_id=worker_id, status="failed", result_json=None, error=error)

    def _finish(self, job_id: str, *, worker_id: str, status: str, result_json: str | None, error: str | None) -> None:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE compute_jobs SET status = ?, result_json = ?, error = ?, lease_owner = NULL, "
                "lease_expires_at = NULL, updated_at = ? WHERE job_id = ? AND lease_owner = ? AND status = 'running'",
                (status, result_json, error, now, job_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("只能完成由当前 worker 运行的任务")

    def cancel(self, job_id: str, *, user_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE compute_jobs SET status = 'cancelled', lease_owner = NULL, lease_expires_at = NULL, "
                "updated_at = ? WHERE job_id = ? AND user_id = ? AND status IN ('queued', 'leased', 'running')",
                (time.time(), job_id, user_id),
            )
            return cursor.rowcount == 1

    def get(self, job_id: str, *, user_id: str) -> ComputeJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM compute_jobs WHERE job_id = ? AND user_id = ?", (job_id, user_id)
            ).fetchone()
        return self._row(row) if row is not None else None

    def get_leased(self, job_id: str, *, worker_id: str) -> ComputeJob | None:
        """供主服务处理 worker 回传物；只接受当前租约持有人。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM compute_jobs WHERE job_id = ? AND lease_owner = ? "
                "AND status = 'running'",
                (job_id, worker_id),
            ).fetchone()
        return self._row(row) if row is not None else None
