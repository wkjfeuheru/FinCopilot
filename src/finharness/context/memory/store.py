"""对话记忆存储：记忆层中持久化的那一半。

一个 SQLite 文件（``settings.paths.memory_db``）保存对话记录（可按 id 检索），
以及在其之上构建的结构化记忆：摘要分段、引用、结论和按对话划分的标的池。

作用域因表而异，这是有意为之：

* 一切以 ``conversation_id`` 为键的内容都是隔离的 —— 一个对话看不到另一个对话
  的记录、摘要、引用或结论；对话之上还有用户作用域（``user_id`` 列），
  因此一个用户也看不到另一个用户的对话；
* ``notes`` 以 ``(user_id, key)`` 为键：用户偏好由该用户的所有对话共享，
  因为“这位用户喜欢什么样的报告”并非某个对话独有，但绝不跨用户共享。

所有语句都使用绑定参数；没有任何一条是由变量拼接而成的。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from finharness.data.citation import Citation
from finharness.types import Msg, ToolUse

SCHEMA_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
)
"""

SCHEMA_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    payload_json TEXT,
    ts TEXT NOT NULL,
    UNIQUE(conversation_id, seq)
)
"""

SCHEMA_SUMMARY_SEGMENTS = """
CREATE TABLE IF NOT EXISTS summary_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    seq_from INTEGER NOT NULL,
    seq_to INTEGER NOT NULL,
    tier INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    ledger_json TEXT,
    created_at TEXT NOT NULL
)
"""

SCHEMA_CITATIONS = """
CREATE TABLE IF NOT EXISTS citations (
    cid TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    symbol TEXT,
    params_json TEXT,
    rows INTEGER NOT NULL DEFAULT 0,
    cols INTEGER NOT NULL DEFAULT 0,
    fingerprint TEXT NOT NULL DEFAULT '',
    parquet_path TEXT,
    from_cache INTEGER NOT NULL DEFAULT 0,
    ts TEXT
)
"""

SCHEMA_CONCLUSIONS = """
CREATE TABLE IF NOT EXISTS conclusions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    text TEXT NOT NULL,
    cids_json TEXT,
    ts TEXT NOT NULL,
    UNIQUE(conversation_id, subject, text)
)
"""

SCHEMA_SYMBOLS = """
CREATE TABLE IF NOT EXISTS conversation_symbols (
    conversation_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    PRIMARY KEY (conversation_id, symbol)
)
"""

SCHEMA_NOTES = """
CREATE TABLE IF NOT EXISTS notes (
    user_id TEXT NOT NULL DEFAULT '',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'preference',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
)
"""

INDEX_MESSAGES_BY_CONVERSATION = (
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, seq)"
)
INDEX_CONCLUSIONS_BY_SUBJECT = (
    "CREATE INDEX IF NOT EXISTS idx_conclusions_subject ON conclusions(conversation_id, subject)"
)
INDEX_CITATIONS_BY_CONVERSATION = (
    "CREATE INDEX IF NOT EXISTS idx_citations_conversation ON citations(conversation_id)"
)
INDEX_CONVERSATIONS_BY_USER = (
    "CREATE INDEX IF NOT EXISTS idx_conversations_user"
    " ON conversations(user_id, last_active_at)"
)


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation_id: str
    created_at: str
    updated_at: str
    last_active_at: str
    user_id: str = ""
    title: str | None = None


@dataclass(frozen=True, slots=True)
class SummarySegment:
    conversation_id: str
    seq_from: int
    seq_to: int
    tier: int
    text: str
    ledger: tuple[str, ...] = ()
    created_at: str = ""
    id: int = 0


@dataclass(frozen=True, slots=True)
class ConclusionRecord:
    conversation_id: str
    subject: str
    text: str
    cids: tuple[str, ...]
    ts: str
    id: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _notes_has_user_scoped_pk(connection: sqlite3.Connection) -> bool:
    """notes 的主键是否为 ``(user_id, key)``。

    ``ON CONFLICT(user_id, key)`` 要求该复合主键存在，因此迁移必须按
    主键组成判定，而不是按 ``user_id`` 列是否存在。
    """
    try:
        rows = connection.execute("PRAGMA table_info(notes)").fetchall()
    except sqlite3.OperationalError:
        return True
    pk_columns = {row["name"] for row in rows if row["pk"]}
    return pk_columns == {"user_id", "key"}


def _encode_payload(message: Msg) -> str | None:
    if not message.tool_uses and not message.tool_results and not message.metadata:
        return None
    return json.dumps(
        {
            "tool_uses": [
                {"call_id": item.call_id, "name": item.name, "args": item.args}
                for item in message.tool_uses
            ],
            "tool_results": [[call_id, raw] for call_id, raw in message.tool_results],
            "metadata": message.metadata,
        },
        ensure_ascii=False,
    )


def _decode_message(row: sqlite3.Row) -> Msg:
    payload = json.loads(row["payload_json"] or "{}")
    tool_uses = [
        ToolUse(call_id=item["call_id"], name=item["name"], args=item.get("args") or {})
        for item in payload.get("tool_uses", [])
    ]
    tool_results = [(pair[0], pair[1]) for pair in payload.get("tool_results", [])]
    return Msg(
        role=row["role"],
        content=row["content"],
        tool_uses=tool_uses,
        tool_results=tool_results,
        metadata=dict(payload.get("metadata") or {}),
    )


class MemoryStore:
    """基于 SQLite 的对话记忆；所有语句都使用绑定参数。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(SCHEMA_CONVERSATIONS)
            connection.execute(SCHEMA_MESSAGES)
            connection.execute(SCHEMA_SUMMARY_SEGMENTS)
            connection.execute(SCHEMA_CITATIONS)
            connection.execute(SCHEMA_CONCLUSIONS)
            connection.execute(SCHEMA_SYMBOLS)
            connection.execute(SCHEMA_NOTES)
            self._migrate(connection)
            connection.execute(INDEX_MESSAGES_BY_CONVERSATION)
            connection.execute(INDEX_CONCLUSIONS_BY_SUBJECT)
            connection.execute(INDEX_CITATIONS_BY_CONVERSATION)
            connection.execute(INDEX_CONVERSATIONS_BY_USER)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """原地升级旧库：补 ``user_id`` 列、把 notes 重建为按用户作用域。

        存量行的 ``user_id`` 为空串，等第一个注册用户认领
        （``claim_user``）；在此之前这些数据不属于任何登录用户。
        """
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(conversations)")
        }
        if "user_id" not in columns:
            connection.execute(
                "ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
            )
        if not _notes_has_user_scoped_pk(connection):
            # SQLite 无法就地改主键：建新表-拷贝-替换。判据是主键组成
            # 而非列是否存在——只检查列会漏掉「有 user_id 但主键仍是 key」
            # 的半迁移状态，那种表的 ON CONFLICT(user_id, key) 会直接报错。
            # 两种旧形态都要兼容：v1 根本没有 user_id 列，半迁移态有。
            legacy_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(notes)")
            }
            legacy_user = "COALESCE(user_id, '')" if "user_id" in legacy_columns else "''"
            connection.execute("ALTER TABLE notes RENAME TO notes_legacy")
            connection.execute(SCHEMA_NOTES)
            connection.execute(
                "INSERT OR IGNORE INTO notes (user_id, key, value, kind, updated_at)"
                f" SELECT {legacy_user}, key, value, kind, updated_at FROM notes_legacy"
            )
            connection.execute("DROP TABLE notes_legacy")

    def claim_user(self, user_id: str) -> int:
        """把无主（``user_id=''``）的对话与笔记划归指定用户。

        单用户时代的存量数据由此被第一个注册的账号继承。返回认领的对话数。
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET user_id = ? WHERE user_id = ''", (user_id,)
            )
            conversations = cursor.rowcount
            connection.execute(
                "UPDATE notes SET user_id = ? WHERE user_id = ''", (user_id,)
            )
        return int(conversations)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    # -- 对话 ------------------------------------------------------------------
    def ensure_conversation(
        self, conversation_id: str, *, user_id: str = "", title: str | None = None
    ) -> ConversationRecord:
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id, user_id, title, created_at, updated_at, last_active_at FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO conversations (conversation_id, user_id, title, created_at, updated_at, last_active_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (conversation_id, user_id, title, now, now, now),
                )
            else:
                # 后续传入的标题只在原标题为空时补上，绝不覆盖。
                connection.execute(
                    "UPDATE conversations SET updated_at = ?, last_active_at = ?, title = COALESCE(title, ?) WHERE conversation_id = ?",
                    (now, now, title, conversation_id),
                )
        record = self.get_conversation(conversation_id)
        assert record is not None
        return record

    def get_conversation(
        self, conversation_id: str, *, user_id: str | None = None
    ) -> ConversationRecord | None:
        """按 id 检索对话；给定 ``user_id`` 时归属不符视同不存在。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id, user_id, title, created_at, updated_at, last_active_at FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None or (user_id is not None and row["user_id"] != user_id):
            return None
        return ConversationRecord(
            conversation_id=row["conversation_id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_active_at=row["last_active_at"],
        )

    def list_conversations(
        self, *, user_id: str | None = None, limit: int = 50
    ) -> list[ConversationRecord]:
        with self._connect() as connection:
            if user_id is None:
                rows = connection.execute(
                    "SELECT conversation_id, user_id, title, created_at, updated_at, last_active_at FROM conversations ORDER BY last_active_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT conversation_id, user_id, title, created_at, updated_at, last_active_at FROM conversations WHERE user_id = ? ORDER BY last_active_at DESC LIMIT ?",
                    (user_id, int(limit)),
                ).fetchall()
        return [
            ConversationRecord(
                conversation_id=row["conversation_id"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_active_at=row["last_active_at"],
            )
            for row in rows
        ]

    def touch_conversation(self, conversation_id: str) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET updated_at = ?, last_active_at = ? WHERE conversation_id = ?",
                (now, now, conversation_id),
            )

    # -- 对话记录 --------------------------------------------------------------
    def append_messages(self, conversation_id: str, messages: list[Msg]) -> tuple[int, int]:
        """在单个事务中写入消息；返回分配到的 seq 区间。"""
        if not messages:
            return (0, 0)
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            base = int(row["max_seq"])
            rows = [
                (
                    conversation_id,
                    base + offset,
                    message.role,
                    message.content,
                    _encode_payload(message),
                    now,
                )
                for offset, message in enumerate(messages, start=1)
            ]
            connection.executemany(
                "INSERT INTO messages (conversation_id, seq, role, content, payload_json, ts) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return (base + 1, base + len(messages))

    def message_seq_range(self, conversation_id: str) -> tuple[int, int]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MIN(seq), 0) AS lo, COALESCE(MAX(seq), 0) AS hi FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return (int(row["lo"]), int(row["hi"]))

    def count_messages(self, conversation_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["n"])

    def load_messages(self, conversation_id: str) -> list[Msg]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content, payload_json FROM messages WHERE conversation_id = ? ORDER BY seq",
                (conversation_id,),
            ).fetchall()
        return [_decode_message(row) for row in rows]

    def attach_latest_answer_metadata(
        self, conversation_id: str, *, metadata: dict, after_seq: int = 0
    ) -> bool:
        """把仅用于回放的元数据挂到最新持久化的最终回答上。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, payload_json FROM messages "
                "WHERE conversation_id = ? AND role = 'assistant' "
                "AND content IS NOT NULL AND content != '' AND seq > ? "
                "ORDER BY seq DESC LIMIT 1",
                (conversation_id, int(after_seq)),
            ).fetchone()
            if row is None:
                return False
            payload = json.loads(row["payload_json"] or "{}")
            payload["metadata"] = {**dict(payload.get("metadata") or {}), **metadata}
            connection.execute(
                "UPDATE messages SET payload_json = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=False), int(row["id"])),
            )
        return True

    def message_by_id(self, message_id: int) -> Msg | None:
        """按行 id 检索单条消息（Q8：内部按 id 检索）。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT role, content, payload_json FROM messages WHERE id = ?",
                (int(message_id),),
            ).fetchone()
        return _decode_message(row) if row is not None else None

    # -- 摘要分段 --------------------------------------------------------------
    def add_summary_segment(
        self,
        conversation_id: str,
        *,
        seq_from: int,
        seq_to: int,
        tier: int,
        text: str,
        ledger: list[str] | None = None,
    ) -> SummarySegment:
        now = _now()
        ledger_json = json.dumps(list(ledger or []), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO summary_segments (conversation_id, seq_from, seq_to, tier, text, ledger_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, int(seq_from), int(seq_to), int(tier), text, ledger_json, now),
            )
        return SummarySegment(
            conversation_id=conversation_id,
            seq_from=int(seq_from),
            seq_to=int(seq_to),
            tier=int(tier),
            text=text,
            ledger=tuple(ledger or []),
            created_at=now,
        )

    def list_summary_segments(self, conversation_id: str) -> list[SummarySegment]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, conversation_id, seq_from, seq_to, tier, text, ledger_json, created_at FROM summary_segments WHERE conversation_id = ? ORDER BY seq_from",
                (conversation_id,),
            ).fetchall()
        segments: list[SummarySegment] = []
        for row in rows:
            segments.append(
                SummarySegment(
                    conversation_id=row["conversation_id"],
                    seq_from=int(row["seq_from"]),
                    seq_to=int(row["seq_to"]),
                    tier=int(row["tier"]),
                    text=row["text"],
                    ledger=tuple(json.loads(row["ledger_json"] or "[]")),
                    created_at=row["created_at"],
                    id=int(row["id"]),
                )
            )
        return segments

    def replace_summary_segments(self, conversation_id: str, segments: list[SummarySegment]) -> None:
        """重写某对话的分段集合，同时保留各段的 seq 区间。"""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM summary_segments WHERE conversation_id = ?",
                (conversation_id,),
            )
            now = _now()
            rows = [
                (
                    conversation_id,
                    int(segment.seq_from),
                    int(segment.seq_to),
                    int(segment.tier),
                    segment.text,
                    json.dumps(list(segment.ledger), ensure_ascii=False),
                    segment.created_at or now,
                )
                for segment in segments
            ]
            if rows:
                connection.executemany(
                    "INSERT INTO summary_segments (conversation_id, seq_from, seq_to, tier, text, ledger_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )

    # -- 引用（cid 必须在重启后原样保留） --------------------------------------
    def save_citations(self, conversation_id: str, citations: list[Citation]) -> None:
        if not citations:
            return
        with self._connect() as connection:
            rows = [
                (
                    item.cid,
                    conversation_id,
                    item.tool,
                    item.endpoint,
                    item.symbol,
                    json.dumps(item.params or {}, ensure_ascii=False, sort_keys=True, default=str),
                    int(item.rows),
                    int(item.cols),
                    item.fingerprint,
                    item.parquet_path,
                    1 if item.from_cache else 0,
                    item.ts,
                )
                for item in citations
            ]
            connection.executemany(
                "INSERT OR REPLACE INTO citations (cid, conversation_id, tool, endpoint, symbol, params_json, rows, cols, fingerprint, parquet_path, from_cache, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def load_citations(self, conversation_id: str) -> list[Citation]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT cid, tool, endpoint, symbol, params_json, rows, cols, fingerprint, parquet_path, from_cache, ts FROM citations WHERE conversation_id = ? ORDER BY cid",
                (conversation_id,),
            ).fetchall()
        citations: list[Citation] = []
        for row in rows:
            citations.append(
                Citation(
                    cid=row["cid"],
                    tool=row["tool"],
                    endpoint=row["endpoint"],
                    symbol=row["symbol"],
                    params=json.loads(row["params_json"] or "{}"),
                    ts=row["ts"] or "",
                    rows=int(row["rows"]),
                    cols=int(row["cols"]),
                    fingerprint=row["fingerprint"] or "",
                    from_cache=bool(row["from_cache"]),
                    parquet_path=row["parquet_path"],
                )
            )
        return citations

    # -- 结论 ------------------------------------------------------------------
    def save_conclusion(
        self, conversation_id: str, *, subject: str, text: str, cids: list[str] | None = None
    ) -> None:
        """幂等：同一 subject 下的同一条结论只存储一次。"""
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conclusions (conversation_id, subject, text, cids_json, ts) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, subject, text, json.dumps(list(cids or []), ensure_ascii=False), _now()),
            )

    def load_conclusions(self, conversation_id: str, *, limit: int = 20) -> list[ConclusionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, conversation_id, subject, text, cids_json, ts FROM conclusions WHERE conversation_id = ? ORDER BY ts DESC LIMIT ?",
                (conversation_id, int(limit)),
            ).fetchall()
        return [
            ConclusionRecord(
                conversation_id=row["conversation_id"],
                subject=row["subject"],
                text=row["text"],
                cids=tuple(json.loads(row["cids_json"] or "[]")),
                ts=row["ts"],
                id=int(row["id"]),
            )
            for row in rows
        ]

    def find_conclusions(self, conversation_id: str, subjects: list[str], *, limit: int = 5) -> list[ConclusionRecord]:
        """按 subject 键召回；每个键一条语句，使 SQL 保持静态。"""
        found: list[ConclusionRecord] = []
        with self._connect() as connection:
            for subject in subjects:
                rows = connection.execute(
                    "SELECT id, conversation_id, subject, text, cids_json, ts FROM conclusions WHERE conversation_id = ? AND subject = ? ORDER BY ts DESC LIMIT ?",
                    (conversation_id, subject, int(limit)),
                ).fetchall()
                for row in rows:
                    found.append(
                        ConclusionRecord(
                            conversation_id=row["conversation_id"],
                            subject=row["subject"],
                            text=row["text"],
                            cids=tuple(json.loads(row["cids_json"] or "[]")),
                            ts=row["ts"],
                            id=int(row["id"]),
                        )
                    )
        return found

    # -- 标的池 ----------------------------------------------------------------
    def upsert_symbol(self, conversation_id: str, symbol: str, *, name: str | None = None) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_symbols (conversation_id, symbol, name, first_seen, last_seen) VALUES (?, ?, ?, ?, ?) ON CONFLICT(conversation_id, symbol) DO UPDATE SET last_seen = excluded.last_seen, name = COALESCE(conversation_symbols.name, excluded.name)",
                (conversation_id, symbol, name, now, now),
            )

    def load_symbols(self, conversation_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT symbol FROM conversation_symbols WHERE conversation_id = ? ORDER BY last_seen DESC",
                (conversation_id,),
            ).fetchall()
        return [row["symbol"] for row in rows]

    # -- 偏好（按用户作用域：该用户的全部对话共享，绝不跨用户） ----------------
    def set_note(self, key: str, value: str, *, user_id: str = "", kind: str = "preference") -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO notes (user_id, key, value, kind, updated_at) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value,"
                " kind = excluded.kind, updated_at = excluded.updated_at",
                (user_id, key, value, kind, _now()),
            )

    def get_notes(self, *, user_id: str = "") -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key, value FROM notes WHERE user_id = ? ORDER BY key", (user_id,)
            ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    # -- 保留策略 --------------------------------------------------------------
    def prune(self, *, max_conversations: int, max_age_days: int) -> list[str]:
        """删除超出任一界限的对话，连同其作用域内的记忆一并清理。

        保留预算按用户独立计算：一个重度用户不应挤掉另一个用户的对话。
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat(timespec="seconds")
        doomed: set[str] = set()
        with self._connect() as connection:
            users = [
                row["user_id"]
                for row in connection.execute(
                    "SELECT DISTINCT user_id FROM conversations"
                ).fetchall()
            ]
            for user in users:
                stale_rows = connection.execute(
                    "SELECT conversation_id FROM conversations WHERE user_id = ? AND last_active_at < ?",
                    (user, cutoff),
                ).fetchall()
                excess_rows = connection.execute(
                    "SELECT conversation_id FROM conversations WHERE user_id = ?"
                    " ORDER BY last_active_at DESC LIMIT -1 OFFSET ?",
                    (user, int(max_conversations)),
                ).fetchall()
                doomed.update(row["conversation_id"] for row in stale_rows)
                doomed.update(row["conversation_id"] for row in excess_rows)
        for conversation_id in doomed:
            self.delete_conversation(conversation_id)
        return sorted(doomed)

    def delete_conversation(self, conversation_id: str) -> None:
        """删除一个对话及其作用域内的全部内容（不含 notebook）。"""
        with self._connect() as connection:
            for statement in (
                "DELETE FROM messages WHERE conversation_id = ?",
                "DELETE FROM summary_segments WHERE conversation_id = ?",
                "DELETE FROM citations WHERE conversation_id = ?",
                "DELETE FROM conclusions WHERE conversation_id = ?",
                "DELETE FROM conversation_symbols WHERE conversation_id = ?",
                "DELETE FROM conversations WHERE conversation_id = ?",
            ):
                connection.execute(statement, (conversation_id,))
