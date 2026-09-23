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

import hashlib
import json
import sqlite3
from array import array
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

# 跨对话长期记忆（docs 03.6.4 LTM）：情节事件——任务结果（每轮结构化写入，
# 无 LLM）、关键决策与对话片段（懒蒸馏产出）。自包含：源对话被 prune 删除
# 不影响已落库的情节（溯源字段冗余了来源标题与时间）。
SCHEMA_LTM_EPISODES = """
CREATE TABLE IF NOT EXISTS ltm_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ep_uid TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT,
    summary TEXT NOT NULL,
    source_conversation_id TEXT,
    source_title TEXT,
    source_ts TEXT,
    cids_json TEXT,
    created_at TEXT NOT NULL,
    distilled INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    UNIQUE(user_id, content_hash)
)
"""

# 语义记忆（docs 03.6.4）：事实/概念/偏好，按 (user_id, key) UPSERT 覆盖。
# 偏好由 notes 表并入（见 _absorb_notes_into_facts），因此不再有独立的偏好机制。
# fa_uid 是 (user_id, key) 的确定性派生 id，与情节的 ep_uid 同为治理寻址口径。
SCHEMA_LTM_FACTS = """
CREATE TABLE IF NOT EXISTS ltm_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fa_uid TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    statement TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    subject TEXT,
    source_conversation_id TEXT,
    source_ts TEXT,
    confidence REAL,
    embedding BLOB,
    embedding_model TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id, key)
)
"""

# 蒸馏台账：记录每个对话的蒸馏进度与失败次数（attempts 超 max 即放弃）。
SCHEMA_LTM_PROCESSED = """
CREATE TABLE IF NOT EXISTS ltm_processed (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    episodes_written INTEGER NOT NULL DEFAULT 0,
    distilled_at TEXT NOT NULL
)
"""

# 轮次级断点（docs 03.3 中断与恢复）。每轮覆盖写一行，因此"最新一行"
# 就是恢复所需的全部状态：消息与引用已由本轮持久化落库，这里只补那些
# 无法从消息记录重建的执行态——首要是研究计划（它只活在 ``ctx`` 里，
# 进程重启即失，而"在同一计划上继续"正是恢复的语义）。``partial_answer``
# 让停止时已确立的内容在客户端重放后依然可见。
SCHEMA_TURN_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS turn_checkpoints (
    conversation_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    reason TEXT,
    rounds INTEGER NOT NULL DEFAULT 0,
    turn_index INTEGER NOT NULL DEFAULT 0,
    plan_json TEXT,
    partial_answer TEXT,
    persisted_seq INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
)
"""

# 断点状态：running 表示一轮已收尾但运行仍在继续（仅作崩溃取证）；
# stopped 表示用户中止或传输断开；completed 表示整轮成功交付。
CHECKPOINT_STATUSES = ("running", "stopped", "completed")

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
INDEX_LTM_EPISODES_BY_USER = (
    "CREATE INDEX IF NOT EXISTS idx_ltm_episodes_user"
    " ON ltm_episodes(user_id, created_at DESC)"
)
INDEX_LTM_EPISODES_BY_SUBJECT = (
    "CREATE INDEX IF NOT EXISTS idx_ltm_episodes_subject"
    " ON ltm_episodes(user_id, subject)"
)
INDEX_LTM_FACTS_BY_KIND = (
    "CREATE INDEX IF NOT EXISTS idx_ltm_facts_kind"
    " ON ltm_facts(user_id, kind)"
)
INDEX_LTM_FACTS_BY_SUBJECT = (
    "CREATE INDEX IF NOT EXISTS idx_ltm_facts_subject"
    " ON ltm_facts(user_id, subject)"
)
INDEX_LTM_FACTS_BY_UID = (
    "CREATE INDEX IF NOT EXISTS idx_ltm_facts_uid ON ltm_facts(fa_uid)"
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


@dataclass(frozen=True, slots=True)
class TurnCheckpoint:
    """一次运行的轮次级断点（docs 03.3）。

    ``recoverable`` 只对 ``stopped`` 为真：那是"用户停止了，可以继续"这一
    状态。``running`` 是崩溃取证用的在途记录（进程若被杀死会留下它），
    ``completed`` 只是收尾标记，两者都不该在界面上提供"继续"入口。
    """

    conversation_id: str
    status: str
    reason: str = ""
    rounds: int = 0
    turn_index: int = 0
    plan: dict | None = None
    partial_answer: str = ""
    persisted_seq: int = 0
    updated_at: str = ""

    @property
    def recoverable(self) -> bool:
        return self.status == "stopped"


# 跨对话情节记忆的类别。task_result 由引擎每轮结构化写入（无 LLM）；
# 其余两类由懒蒸馏产出。
LTM_EPISODE_KINDS = ("task_result", "decision", "excerpt")
# 跨对话语义记忆的类别（docs 03.6.4）：事实、概念、偏好。
LTM_FACT_KINDS = ("fact", "concept", "preference")
# ltm_episodes.summary 的长度上限：情节刻意保持"一行摘要 + 指针"的体量，
# 防止 LTM 变成第二个上下文窗口（与 L2 Episode 的纪律一致）。
LTM_MAX_SUMMARY_CHARS = 400
# 语义条目的长度上限；比情节略宽，因为一条"事实"往往需要完整表述。
LTM_MAX_STATEMENT_CHARS = 500


@dataclass(frozen=True, slots=True)
class LtmEpisodeRecord:
    """一条跨对话情节：自包含（带溯源），按用户作用域。"""

    kind: str
    summary: str
    subject: str = ""
    cids: tuple[str, ...] = ()
    source_conversation_id: str = ""
    source_title: str = ""
    source_ts: str = ""
    created_at: str = ""
    ep_uid: str = ""
    distilled: bool = False
    id: int = 0


@dataclass(frozen=True, slots=True)
class LtmFactRecord:
    """一条跨对话语义记忆：事实 / 概念 / 偏好，按 ``(user_id, key)`` 唯一。

    ``fa_uid`` 由 (user, key) 确定性派生，因此它与 ``key`` 一样稳定可寻址——
    治理（编辑/删除）与工具寻址都用它，与情节的 ``ep_uid`` 同一形态。
    """

    key: str
    statement: str
    kind: str = "fact"
    subject: str = ""
    source_conversation_id: str = ""
    source_ts: str = ""
    confidence: float | None = None
    updated_at: str = ""
    fa_uid: str = ""
    has_embedding: bool = False
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
            connection.execute(SCHEMA_LTM_EPISODES)
            connection.execute(SCHEMA_LTM_FACTS)
            connection.execute(SCHEMA_LTM_PROCESSED)
            connection.execute(SCHEMA_TURN_CHECKPOINTS)
            self._migrate(connection)
            connection.execute(INDEX_MESSAGES_BY_CONVERSATION)
            connection.execute(INDEX_CONCLUSIONS_BY_SUBJECT)
            connection.execute(INDEX_CITATIONS_BY_CONVERSATION)
            connection.execute(INDEX_CONVERSATIONS_BY_USER)
            connection.execute(INDEX_LTM_EPISODES_BY_USER)
            connection.execute(INDEX_LTM_EPISODES_BY_SUBJECT)
            connection.execute(INDEX_LTM_FACTS_BY_KIND)
            connection.execute(INDEX_LTM_FACTS_BY_SUBJECT)
            connection.execute(INDEX_LTM_FACTS_BY_UID)
            self._absorb_notes_into_facts(connection)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """原地升级旧库：补 ``user_id`` 列、把 notes 重建为按用户作用域、
        给 ``ltm_facts`` 补 ``fa_uid`` 列。

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
        fact_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(ltm_facts)")
        }
        if fact_columns and "fa_uid" not in fact_columns:
            # 一期只建了表、没写过行，但仍按正式迁移处理：补列并回填派生 id。
            connection.execute(
                "ALTER TABLE ltm_facts ADD COLUMN fa_uid TEXT NOT NULL DEFAULT ''"
            )
            rows = connection.execute(
                "SELECT id, user_id, key FROM ltm_facts WHERE fa_uid = ''"
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE ltm_facts SET fa_uid = ? WHERE id = ?",
                    (self._fa_uid(row["user_id"], row["key"]), int(row["id"])),
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
        """把无主（``user_id=''``）的对话、笔记与长期记忆划归指定用户。

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
            # 跨对话情节与蒸馏台账同样认领。情节带 (user_id, content_hash) 唯一
            # 约束，因此先删掉与目标用户已有情节重复的无主行，再整体改写归属。
            connection.execute(
                "DELETE FROM ltm_episodes WHERE user_id = '' AND EXISTS ("
                " SELECT 1 FROM ltm_episodes claimed WHERE claimed.user_id = ?"
                " AND claimed.content_hash = ltm_episodes.content_hash)",
                (user_id,),
            )
            connection.execute(
                "UPDATE ltm_episodes SET user_id = ? WHERE user_id = ''", (user_id,)
            )
            connection.execute(
                "UPDATE ltm_processed SET user_id = ? WHERE user_id = ''", (user_id,)
            )
            # 语义条目同理：先删重复键的无主行，再改写归属并重算派生 id。
            connection.execute(
                "DELETE FROM ltm_facts WHERE user_id = '' AND EXISTS ("
                " SELECT 1 FROM ltm_facts claimed WHERE claimed.user_id = ?"
                " AND claimed.key = ltm_facts.key)",
                (user_id,),
            )
            connection.execute(
                "UPDATE ltm_facts SET user_id = ? WHERE user_id = ''", (user_id,)
            )
            rows = connection.execute(
                "SELECT id, user_id, key FROM ltm_facts WHERE user_id = ?"
                " AND (fa_uid = '' OR fa_uid IS NULL)",
                (user_id,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE ltm_facts SET fa_uid = ? WHERE id = ?",
                    (self._fa_uid(row["user_id"], row["key"]), int(row["id"])),
                )
        return int(conversations)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def ping(self) -> None:
        """确认记忆数据库可建立连接并执行查询。"""
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()

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

    def conversation_stats_by_user(self) -> dict[str, dict[str, str | int]]:
        """每用户的对话数与最后活跃时间（管理员页用户总览数据源）。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, COUNT(*) AS conversations, MAX(last_active_at) AS last_active_at"
                " FROM conversations GROUP BY user_id"
            ).fetchall()
        return {
            row["user_id"]: {
                "conversations": int(row["conversations"]),
                "last_active_at": row["last_active_at"] or "",
            }
            for row in rows
            if row["user_id"]
        }

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

    def load_messages(
        self, conversation_id: str, *, recent_rounds: int | None = None
    ) -> list[Msg]:
        """读取对话记录；``recent_rounds`` 限定时只回放最近若干轮。

        长对话的完整记录可能有几百轮，而每轮都带着工具结果。全部读进内存会让
        会话重载的峰值等于整份历史；更早的历史其实已由 ``summary_segments``
        承载，不必逐字重放。

        切点必须落在 assistant 帧上：一个被保留的 tool_result，若其 call_id
        从未被某条 assistant 消息声明过，会被 OpenAI 兼容的 API 拒绝。因此这里
        按 assistant 帧倒数定位，而不是按消息偏移。
        """
        sql = "SELECT role, content, payload_json FROM messages WHERE conversation_id = ?"
        params: list[object] = [conversation_id]
        if recent_rounds is not None and recent_rounds > 0:
            with self._connect() as connection:
                anchor = connection.execute(
                    "SELECT seq FROM messages WHERE conversation_id = ? AND role = 'assistant' "
                    "ORDER BY seq DESC LIMIT 1 OFFSET ?",
                    (conversation_id, int(recent_rounds) - 1),
                ).fetchone()
            if anchor is not None:
                # 该助手帧之后（含）的全部消息。它的 seq 保证是某轮的起点。
                sql += " AND seq >= ?"
                params.append(int(anchor["seq"]))
        sql += " ORDER BY seq"
        with self._connect() as connection:
            rows = connection.execute(sql, tuple(params)).fetchall()
        return [_decode_message(row) for row in rows]

    def count_covered_prefix(self, conversation_id: str) -> int:
        """已被摘要分段覆盖的消息序号上限（从 1 起连续覆盖到的位置）。

        有界加载必须知道"被跳过的前缀是否已有摘要兜底"，否则会静默丢掉没人
        记得的历史。返回 0 表示没有任何分段覆盖前缀。
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT seq_from, seq_to FROM summary_segments "
                "WHERE conversation_id = ? ORDER BY seq_from",
                (conversation_id,),
            ).fetchall()
        covered = 0
        for row in rows:
            start = int(row["seq_from"])
            end = int(row["seq_to"])
            # 只认从首条消息开始、且连续的覆盖；中间有洞就等于没覆盖到洞之后。
            if start > covered + 1:
                break
            covered = max(covered, end)
        return covered

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

    # -- 轮次级断点（docs 03.3 中断与恢复） ------------------------------------
    def save_checkpoint(
        self,
        conversation_id: str,
        *,
        status: str,
        reason: str = "",
        rounds: int = 0,
        turn_index: int = 0,
        plan: dict | None = None,
        partial_answer: str = "",
        persisted_seq: int = 0,
    ) -> TurnCheckpoint:
        """写入/覆盖该对话的最新断点（按 conversation_id 主键，最新覆盖）。

        每轮覆盖而非追加：恢复只需要最新状态，历史断点既无消费者也会无界增长。
        ``plan`` 为空时写 NULL，使它被区分于"有一份空计划"。
        """
        if status not in CHECKPOINT_STATUSES:
            raise ValueError(
                f"unknown checkpoint status: {status}; expected one of {CHECKPOINT_STATUSES}"
            )
        now = _now()
        plan_json = json.dumps(plan, ensure_ascii=False) if plan else None
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO turn_checkpoints ("
                " conversation_id, status, reason, rounds, turn_index,"
                " plan_json, partial_answer, persisted_seq, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(conversation_id) DO UPDATE SET"
                " status = excluded.status, reason = excluded.reason,"
                " rounds = excluded.rounds, turn_index = excluded.turn_index,"
                " plan_json = excluded.plan_json,"
                " partial_answer = excluded.partial_answer,"
                " persisted_seq = excluded.persisted_seq,"
                " updated_at = excluded.updated_at",
                (
                    conversation_id,
                    status,
                    reason or None,
                    int(rounds),
                    int(turn_index),
                    plan_json,
                    partial_answer or None,
                    int(persisted_seq),
                    now,
                ),
            )
        return TurnCheckpoint(
            conversation_id=conversation_id,
            status=status,
            reason=reason or "",
            rounds=int(rounds),
            turn_index=int(turn_index),
            plan=plan,
            partial_answer=partial_answer or "",
            persisted_seq=int(persisted_seq),
            updated_at=now,
        )

    def load_latest_checkpoint(self, conversation_id: str) -> TurnCheckpoint | None:
        """该对话的最新断点；从未写过返回 ``None``。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id, status, reason, rounds, turn_index,"
                " plan_json, partial_answer, persisted_seq, updated_at"
                " FROM turn_checkpoints WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        plan = json.loads(row["plan_json"]) if row["plan_json"] else None
        return TurnCheckpoint(
            conversation_id=row["conversation_id"],
            status=row["status"],
            reason=row["reason"] or "",
            rounds=int(row["rounds"] or 0),
            turn_index=int(row["turn_index"] or 0),
            plan=plan if isinstance(plan, dict) else None,
            partial_answer=row["partial_answer"] or "",
            persisted_seq=int(row["persisted_seq"] or 0),
            updated_at=row["updated_at"] or "",
        )

    def clear_checkpoint(self, conversation_id: str) -> None:
        """清除某对话的断点（例如新一轮重新提问后，旧的"可继续"状态失效）。"""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM turn_checkpoints WHERE conversation_id = ?",
                (conversation_id,),
            )

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

    # -- 跨对话情节记忆（docs 03.6.4 LTM，按用户作用域） ----------------------
    def _ltm_episode_hash(self, *, kind: str, subject: str, summary: str) -> str:
        """情节去重键：同一用户对同一 (kind, subject, summary) 只记一次。"""
        digest = hashlib.sha256(
            "|".join((kind, subject.strip(), summary.strip())).encode("utf-8")
        )
        return digest.hexdigest()[:32]

    def _clip_ltm_summary(self, summary: str) -> str:
        text = (summary or "").strip()
        if len(text) > LTM_MAX_SUMMARY_CHARS:
            text = text[: LTM_MAX_SUMMARY_CHARS - 1].rstrip() + "…"
        return text

    def add_ltm_episode(
        self,
        *,
        kind: str,
        summary: str,
        user_id: str = "",
        subject: str = "",
        cids: list[str] | None = None,
        source_conversation_id: str = "",
        source_title: str = "",
        source_ts: str = "",
        distilled: bool = False,
    ) -> LtmEpisodeRecord | None:
        """幂等写入一条情节；内容重复（同 hash）时返回 ``None``。

        ``ep_uid`` 由内容哈希派生，因此对外稳定且可据以检索/删除；
        同一情节的重复写入（每轮结构化写入与懒蒸馏重叠）自然去重。
        """
        if kind not in LTM_EPISODE_KINDS:
            raise ValueError(f"unknown LTM episode kind: {kind}; expected {LTM_EPISODE_KINDS}")
        text = self._clip_ltm_summary(summary)
        if not text:
            return None
        subject = (subject or "").strip()
        content_hash = self._ltm_episode_hash(kind=kind, subject=subject, summary=text)
        # 对外 id 由 (用户, 内容) 派生：不同用户写入相同内容时各自拥有独立的
        # ep_uid（去重是用户作用域的），而同一用户重写同一内容仍幂等。
        uid_digest = hashlib.sha256(
            f"{user_id}|{content_hash}".encode("utf-8")
        ).hexdigest()[:12]
        ep_uid = f"ep_{uid_digest}"
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO ltm_episodes"
                " (ep_uid, user_id, kind, subject, summary, source_conversation_id,"
                " source_title, source_ts, cids_json, created_at, distilled, content_hash)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ep_uid,
                    user_id,
                    kind,
                    subject,
                    text,
                    source_conversation_id,
                    source_title,
                    source_ts,
                    json.dumps(list(cids or []), ensure_ascii=False),
                    now,
                    1 if distilled else 0,
                    content_hash,
                ),
            )
            if cursor.rowcount == 0:
                return None
        return LtmEpisodeRecord(
            kind=kind,
            summary=text,
            subject=subject,
            cids=tuple(cids or []),
            source_conversation_id=source_conversation_id,
            source_title=source_title,
            source_ts=source_ts,
            created_at=now,
            ep_uid=ep_uid,
            distilled=distilled,
        )

    def _row_to_ltm_episode(self, row: sqlite3.Row) -> LtmEpisodeRecord:
        return LtmEpisodeRecord(
            kind=row["kind"],
            summary=row["summary"],
            subject=row["subject"] or "",
            cids=tuple(json.loads(row["cids_json"] or "[]")),
            source_conversation_id=row["source_conversation_id"] or "",
            source_title=row["source_title"] or "",
            source_ts=row["source_ts"] or "",
            created_at=row["created_at"],
            ep_uid=row["ep_uid"],
            distilled=bool(row["distilled"]),
            id=int(row["id"]),
        )

    _LTM_COLUMNS = (
        "id, ep_uid, user_id, kind, subject, summary, source_conversation_id,"
        " source_title, source_ts, cids_json, created_at, distilled"
    )

    def list_ltm_episodes(
        self,
        *,
        user_id: str = "",
        subject: str | None = None,
        kind: str | None = None,
        source_conversation_id: str | None = None,
        since: str | None = None,
        limit: int = 20,
    ) -> list[LtmEpisodeRecord]:
        """按结构化字段检索情节；最新优先。所有过滤都是键匹配，无向量。"""
        clauses = ["user_id = ?"]
        params: list[object] = [user_id]
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if source_conversation_id is not None:
            clauses.append("source_conversation_id = ?")
            params.append(source_conversation_id)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        sql = (
            f"SELECT {self._LTM_COLUMNS} FROM ltm_episodes"
            f" WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, id DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(sql, tuple(params)).fetchall()
        return [self._row_to_ltm_episode(row) for row in rows]

    def get_ltm_episode(
        self, ep_uid: str, *, user_id: str | None = None
    ) -> LtmEpisodeRecord | None:
        """按对外 id 取单条情节；``user_id`` 给定时归属不符视同不存在。"""
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {self._LTM_COLUMNS} FROM ltm_episodes WHERE ep_uid = ?",
                (ep_uid,),
            ).fetchone()
        if row is None or (user_id is not None and row["user_id"] != user_id):
            return None
        return self._row_to_ltm_episode(row)

    def update_ltm_episode(
        self, ep_uid: str, *, user_id: str, kind: str | None = None,
        subject: str | None = None, summary: str | None = None,
    ) -> LtmEpisodeRecord | None:
        """编辑一条情节（治理）；未命中的 id / 归属不符返回 ``None``。

        summary 改动会重算去重键：编辑成与既有情节相同的内容会被唯一约束
        拒绝（抛 sqlite3.IntegrityError），由调用方转成 409。
        """
        existing = self.get_ltm_episode(ep_uid, user_id=user_id)
        if existing is None:
            return None
        new_kind = kind if kind is not None else existing.kind
        if new_kind not in LTM_EPISODE_KINDS:
            raise ValueError(f"unknown LTM episode kind: {new_kind}")
        new_subject = subject if subject is not None else existing.subject
        new_summary = (
            self._clip_ltm_summary(summary) if summary is not None else existing.summary
        )
        content_hash = self._ltm_episode_hash(
            kind=new_kind, subject=new_subject, summary=new_summary
        )
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE ltm_episodes SET kind = ?, subject = ?, summary = ?,"
                " content_hash = ? WHERE ep_uid = ? AND user_id = ?",
                (new_kind, new_subject, new_summary, content_hash, ep_uid, user_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_ltm_episode(ep_uid, user_id=user_id)

    def delete_ltm_episode(self, ep_uid: str, *, user_id: str) -> bool:
        """删除一条情节（治理）；未命中或归属不符返回 False。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM ltm_episodes WHERE ep_uid = ? AND user_id = ?",
                (ep_uid, user_id),
            )
            return cursor.rowcount > 0

    def prune_ltm_episodes(
        self, *, user_id: str, max_episodes: int, max_age_days: int
    ) -> int:
        """情节的独立保留：超出条数或天数上限时删最旧，返回删除数。

        与源对话的 prune 无关：情节自包含，源对话被删不影响这里。按用户
        独立计算，一个重度用户不挤占另一个的长期记忆。
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
        ).isoformat(timespec="seconds")
        with self._connect() as connection:
            stale = connection.execute(
                "SELECT ep_uid FROM ltm_episodes WHERE user_id = ? AND created_at < ?",
                (user_id, cutoff),
            ).fetchall()
            doomed = {row["ep_uid"] for row in stale}
            excess = connection.execute(
                "SELECT ep_uid FROM ltm_episodes WHERE user_id = ?"
                " ORDER BY created_at DESC, id DESC LIMIT -1 OFFSET ?",
                (user_id, int(max_episodes)),
            ).fetchall()
            doomed.update(row["ep_uid"] for row in excess)
            for ep_uid in sorted(doomed):
                connection.execute(
                    "DELETE FROM ltm_episodes WHERE user_id = ? AND ep_uid = ?",
                    (user_id, ep_uid),
                )
        return len(doomed)

    # -- 蒸馏台账 --------------------------------------------------------------
    def mark_ltm_distilled(
        self, conversation_id: str, *, user_id: str, episodes_written: int = 0, attempts: int | None = None
    ) -> None:
        """记录一次蒸馏结果；``attempts`` 给定时累加失败计数。"""
        now = _now()
        with self._connect() as connection:
            if attempts is None:
                connection.execute(
                    "INSERT INTO ltm_processed (conversation_id, user_id, attempts, episodes_written, distilled_at)"
                    " VALUES (?, ?, 0, ?, ?)"
                    " ON CONFLICT(conversation_id) DO UPDATE SET"
                    " episodes_written = excluded.episodes_written, distilled_at = excluded.distilled_at",
                    (conversation_id, user_id, episodes_written, now),
                )
            else:
                connection.execute(
                    "INSERT INTO ltm_processed (conversation_id, user_id, attempts, episodes_written, distilled_at)"
                    " VALUES (?, ?, ?, 0, ?)"
                    " ON CONFLICT(conversation_id) DO UPDATE SET attempts = excluded.attempts",
                    (conversation_id, user_id, attempts, now),
                )

    def get_ltm_distill_state(self, conversation_id: str) -> tuple[int, int] | None:
        """返回 (attempts, episodes_written)；无记录返回 ``None``。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT attempts, episodes_written FROM ltm_processed WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return (int(row["attempts"]), int(row["episodes_written"]))

    def ltm_distill_candidates(
        self, *, user_id: str, exclude_conversation: str | None = None,
        max_attempts: int, idle_before: str, limit: int,
    ) -> list[str]:
        """待蒸馏对话：属于该用户、闲置超时、无成功记录且未达重试上限。

        闲置判定读 ``conversations.last_active_at``（而非会话注册表——
        那是进程内的执行窗口，与"对话多久没人用"是两回事）。
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT c.conversation_id FROM conversations c"
                " LEFT JOIN ltm_processed p ON p.conversation_id = c.conversation_id"
                " WHERE c.user_id = ? AND c.last_active_at < ?"
                " AND (p.conversation_id IS NULL OR (p.episodes_written = 0 AND p.attempts < ?))"
                " ORDER BY c.last_active_at ASC LIMIT ?",
                (user_id, idle_before, int(max_attempts), int(limit)),
            ).fetchall()
        ids = [row["conversation_id"] for row in rows]
        if exclude_conversation:
            ids = [cid for cid in ids if cid != exclude_conversation]
        return ids

    def ltm_users_with_candidates(
        self, *, max_attempts: int, idle_before: str, limit: int
    ) -> list[str]:
        """存在待蒸馏对话的用户列表（供服务端后台扫描器使用）。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT c.user_id FROM conversations c"
                " LEFT JOIN ltm_processed p ON p.conversation_id = c.conversation_id"
                " WHERE c.last_active_at < ?"
                " AND (p.conversation_id IS NULL OR (p.episodes_written = 0 AND p.attempts < ?))"
                " ORDER BY c.last_active_at ASC LIMIT ?",
                (idle_before, int(max_attempts), int(limit)),
            ).fetchall()
        return [row["user_id"] for row in rows]

    def prune_all_ltm_episodes(self, *, max_episodes: int, max_age_days: int) -> int:
        """对所有有情节的用户执行一次 LTM 保留；返回删除总数。

        与 ``prune`` 并列在服务启动时调用：蒸馏后的保留（``_prune``）只在
        真的发生蒸馏时触发，而 task_result 情节每轮写入、不依赖蒸馏，因此
        需要这条兜底路径，否则只跑不蒸馏的用户其 LTM 会无界增长。
        """
        with self._connect() as connection:
            users = [
                row["user_id"]
                for row in connection.execute(
                    "SELECT DISTINCT user_id FROM ltm_episodes"
                ).fetchall()
            ]
        return sum(
            self.prune_ltm_episodes(
                user_id=user, max_episodes=max_episodes, max_age_days=max_age_days
            )
            for user in users
        )

    # -- 跨对话语义记忆（docs 03.6.4 LTM，按用户作用域） ----------------------
    @staticmethod
    def _fa_uid(user_id: str, key: str) -> str:
        """语义条目的对外 id：由 (user, key) 确定性派生。"""
        digest = hashlib.sha256(f"{user_id}|{key}".encode("utf-8")).hexdigest()[:12]
        return f"fa_{digest}"

    @staticmethod
    def encode_vector(vector: list[float]) -> bytes:
        """向量按 4 字节浮点编码为 BLOB（无向量库时的本地检索路径）。"""
        return array("f", [float(value) for value in vector]).tobytes()

    @staticmethod
    def decode_vector(blob: bytes | None) -> list[float]:
        if not blob:
            return []
        values = array("f")
        values.frombytes(blob)
        return list(values)

    def _row_to_ltm_fact(self, row: sqlite3.Row) -> LtmFactRecord:
        return LtmFactRecord(
            key=row["key"],
            statement=row["statement"],
            kind=row["kind"],
            subject=row["subject"] or "",
            source_conversation_id=row["source_conversation_id"] or "",
            source_ts=row["source_ts"] or "",
            confidence=row["confidence"],
            updated_at=row["updated_at"],
            fa_uid=row["fa_uid"] or self._fa_uid(row["user_id"], row["key"]),
            has_embedding=row["embedding"] is not None,
            id=int(row["id"]),
        )

    _LTM_FACT_COLUMNS = (
        "id, fa_uid, user_id, key, statement, kind, subject, source_conversation_id,"
        " source_ts, confidence, embedding, updated_at"
    )

    def upsert_ltm_fact(
        self,
        *,
        user_id: str = "",
        key: str,
        statement: str,
        kind: str = "fact",
        subject: str = "",
        source_conversation_id: str = "",
        source_ts: str = "",
        confidence: float | None = None,
    ) -> LtmFactRecord | None:
        """按 ``(user_id, key)`` UPSERT 语义条目——覆盖语义，不是追加。

        与情节的"只增不去重"相反：一条事实的更新版本应当**替换**旧表述，
        否则注入时会出现互相矛盾的两条。偏好尤其如此——用户后来明确说的
        口径必须赢过蒸馏出的旧口径，这与"冲突直接覆盖"的治理约定一致。

        改 ``statement`` 会让已存的向量失效，因此顺带清空 ``embedding``，
        由嵌入回填路径重算（见 ``facts_missing_embedding``）。
        """
        if kind not in LTM_FACT_KINDS:
            raise ValueError(f"unknown LTM fact kind: {kind}; expected {LTM_FACT_KINDS}")
        clean_key = (key or "").strip()
        text = (statement or "").strip()
        if not clean_key or not text:
            return None
        if len(text) > LTM_MAX_STATEMENT_CHARS:
            text = text[: LTM_MAX_STATEMENT_CHARS - 1].rstrip() + "…"
        now = _now()
        fa_uid = self._fa_uid(user_id, clean_key)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, statement FROM ltm_facts WHERE user_id = ? AND key = ?",
                (user_id, clean_key),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO ltm_facts"
                    " (fa_uid, user_id, key, statement, kind, subject,"
                    " source_conversation_id, source_ts, confidence, embedding,"
                    " embedding_model, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)",
                    (
                        fa_uid, user_id, clean_key, text, kind, subject,
                        source_conversation_id, source_ts, confidence, now,
                    ),
                )
            else:
                same = existing["statement"] == text
                connection.execute(
                    "UPDATE ltm_facts SET statement = ?, kind = ?, subject = ?,"
                    " source_conversation_id = ?, source_ts = ?,"
                    " confidence = COALESCE(?, confidence), updated_at = ?"
                    # 表述未变则保留向量；变了就置空，等回填重算。
                    + ("" if same else ", embedding = NULL, embedding_model = NULL")
                    + " WHERE user_id = ? AND key = ?",
                    (
                        text, kind, subject, source_conversation_id, source_ts,
                        confidence, now, user_id, clean_key,
                    ),
                )
        return self.get_ltm_fact_by_key(user_id=user_id, key=clean_key)

    def get_ltm_fact(self, fa_uid: str, *, user_id: str | None = None) -> LtmFactRecord | None:
        """按对外 id 取单条语义条目；``user_id`` 给定时归属不符视同不存在。"""
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {self._LTM_FACT_COLUMNS} FROM ltm_facts WHERE fa_uid = ?",
                (fa_uid,),
            ).fetchone()
        if row is None or (user_id is not None and row["user_id"] != user_id):
            return None
        return self._row_to_ltm_fact(row)

    def get_ltm_fact_by_key(self, *, user_id: str, key: str) -> LtmFactRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {self._LTM_FACT_COLUMNS} FROM ltm_facts"
                " WHERE user_id = ? AND key = ?",
                (user_id, key),
            ).fetchone()
        return self._row_to_ltm_fact(row) if row is not None else None

    def list_ltm_facts(
        self,
        *,
        user_id: str = "",
        kind: str | None = None,
        subject: str | None = None,
        limit: int = 50,
    ) -> list[LtmFactRecord]:
        """按键检索语义条目；偏好优先（注入顺序需要它），其余按更新时间倒序。"""
        clauses = ["user_id = ?"]
        params: list[object] = [user_id]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        sql = (
            f"SELECT {self._LTM_FACT_COLUMNS} FROM ltm_facts"
            f" WHERE {' AND '.join(clauses)}"
            " ORDER BY updated_at DESC, id DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(sql, tuple(params)).fetchall()
        return [self._row_to_ltm_fact(row) for row in rows]

    def update_ltm_fact(
        self,
        fa_uid: str,
        *,
        user_id: str,
        statement: str | None = None,
        kind: str | None = None,
        subject: str | None = None,
    ) -> LtmFactRecord | None:
        """编辑一条语义条目（治理）；未命中或归属不符返回 ``None``。"""
        existing = self.get_ltm_fact(fa_uid, user_id=user_id)
        if existing is None:
            return None
        return self.upsert_ltm_fact(
            user_id=user_id,
            key=existing.key,
            statement=statement if statement is not None else existing.statement,
            kind=kind if kind is not None else existing.kind,
            subject=subject if subject is not None else existing.subject,
            source_conversation_id=existing.source_conversation_id,
            source_ts=existing.source_ts,
        )

    def delete_ltm_fact(self, fa_uid: str, *, user_id: str) -> bool:
        """删除一条语义条目（治理）；未命中或归属不符返回 False。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM ltm_facts WHERE fa_uid = ? AND user_id = ?",
                (fa_uid, user_id),
            )
            return cursor.rowcount > 0

    def delete_ltm_fact_by_key(self, *, user_id: str, key: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM ltm_facts WHERE user_id = ? AND key = ?", (user_id, key)
            )
            return cursor.rowcount > 0

    def set_ltm_fact_embedding(
        self, *, user_id: str, key: str, vector: list[float], model: str = ""
    ) -> bool:
        """写入语义条目的向量（本地 BLOB 副本；无向量库时的检索路径）。"""
        if not vector:
            return False
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE ltm_facts SET embedding = ?, embedding_model = ?"
                " WHERE user_id = ? AND key = ?",
                (self.encode_vector(vector), model, user_id, key),
            )
            return cursor.rowcount > 0

    def facts_missing_embedding(
        self, *, user_id: str = "", limit: int = 50
    ) -> list[LtmFactRecord]:
        """尚未向量化的条目（新写入的、或被改写后向量失效的）。"""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {self._LTM_FACT_COLUMNS} FROM ltm_facts"
                " WHERE user_id = ? AND embedding IS NULL"
                " ORDER BY updated_at DESC LIMIT ?",
                (user_id, int(limit)),
            ).fetchall()
        return [self._row_to_ltm_fact(row) for row in rows]

    def list_fact_vectors(self, *, user_id: str = "") -> list[tuple[str, list[float]]]:
        """该用户全部已向量化条目：``[(fa_uid, vector), ...]``。

        供无向量库时的本地暴力余弦检索使用——Qdrant 不可用不该等于没有语义
        召回，这条路径让"只配了 embedding 端点"的部署同样能工作。
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, key, embedding FROM ltm_facts"
                " WHERE user_id = ? AND embedding IS NOT NULL",
                (user_id,),
            ).fetchall()
        return [
            (self._fa_uid(row["user_id"], row["key"]), self.decode_vector(row["embedding"]))
            for row in rows
        ]

    def prune_ltm_facts(
        self, *, user_id: str, max_facts: int, max_age_days: int
    ) -> int:
        """语义条目的独立保留：超条数/天数上限时删最旧，返回删除数。

        偏好永不因条数上限被删——它们是用户显式设定或长期稳定的口径，
        被自动清理掉会让"记住我的偏好"变成一句空话。
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
        ).isoformat(timespec="seconds")
        with self._connect() as connection:
            stale = connection.execute(
                "SELECT user_id, key FROM ltm_facts"
                " WHERE user_id = ? AND kind != 'preference' AND updated_at < ?",
                (user_id, cutoff),
            ).fetchall()
            excess = connection.execute(
                "SELECT user_id, key FROM ltm_facts"
                " WHERE user_id = ? AND kind != 'preference'"
                " ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?",
                (user_id, int(max_facts)),
            ).fetchall()
            doomed = {(row["user_id"], row["key"]) for row in stale}
            doomed.update((row["user_id"], row["key"]) for row in excess)
            for owner, key in sorted(doomed):
                connection.execute(
                    "DELETE FROM ltm_facts WHERE user_id = ? AND key = ?", (owner, key)
                )
        return len(doomed)

    def prune_all_ltm_facts(self, *, max_facts: int, max_age_days: int) -> int:
        """对所有有语义条目的用户执行一次保留；返回删除总数。"""
        with self._connect() as connection:
            users = [
                row["user_id"]
                for row in connection.execute(
                    "SELECT DISTINCT user_id FROM ltm_facts"
                ).fetchall()
            ]
        return sum(
            self.prune_ltm_facts(
                user_id=user, max_facts=max_facts, max_age_days=max_age_days
            )
            for user in users
        )

    def ltm_fact_users(self) -> list[str]:
        """拥有语义条目的用户列表（供向量回填逐用户执行）。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT user_id FROM ltm_facts"
            ).fetchall()
        return [row["user_id"] for row in rows]

    # -- 偏好（吸收自 notes 表；读写都落在 ltm_facts） --------------------------
    def _absorb_notes_into_facts(self, connection: sqlite3.Connection) -> None:
        """把旧 ``notes`` 表的内容一次性并入 ``ltm_facts``。

        语义记忆统一到一张表后，偏好不再是独立机制：``(user_id, key)`` 的
        UPSERT 语义本就能表达"覆盖"，且它天然带有溯源与向量字段。迁移是
        ``INSERT OR IGNORE``，因此可重复执行；已存在同键条目时以 facts 为准，
        绝不把用户后来的更新用旧值盖回去。
        """
        try:
            connection.execute(
                "INSERT OR IGNORE INTO ltm_facts"
                " (user_id, key, statement, kind, subject, source_conversation_id,"
                "  source_ts, confidence, embedding, embedding_model, updated_at)"
                " SELECT user_id, key, value,"
                "  CASE WHEN kind IN ('fact', 'concept', 'preference')"
                "       THEN kind ELSE 'preference' END,"
                "  '', '', '', NULL, NULL, NULL, updated_at"
                " FROM notes WHERE key != '' AND value != ''"
            )
        except sqlite3.OperationalError:
            # 旧库里没有 notes 表（全新安装）：迁移无事可做。
            pass

    def set_note(
        self, key: str, value: str, *, user_id: str = "", kind: str = "preference"
    ) -> None:
        """写入一条用户偏好（``ltm_facts``，``(user_id, key)`` 覆盖语义）。

        保留此方法名与签名：``remember_preference`` 工具与既有测试都按它调用，
        而底层已统一到语义表——偏好与蒸馏出的条目因此共享溯源、去重与检索。
        """
        self.upsert_ltm_fact(user_id=user_id, key=key, statement=value, kind=kind)

    def get_notes(self, *, user_id: str = "") -> dict[str, str]:
        """该用户的全部偏好，形如 ``{key: value}``（原 notes 表的对外形状）。"""
        return {
            fact.key: fact.statement
            for fact in self.list_ltm_facts(
                user_id=user_id, kind="preference", limit=1000
            )
        }

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
        """删除一个对话及其作用域内的全部内容（不含 notebook）。

        刻意**不**删除 ``ltm_episodes``：跨对话情节是自包含的（冗余了来源
        标题与时间，并用 cid 指向引用），因此源对话被删不该抹掉已经沉淀的
        长期记忆——它由 ``prune_ltm_episodes`` 按自己的预算管理。
        """
        with self._connect() as connection:
            for statement in (
                "DELETE FROM messages WHERE conversation_id = ?",
                "DELETE FROM summary_segments WHERE conversation_id = ?",
                "DELETE FROM citations WHERE conversation_id = ?",
                "DELETE FROM conclusions WHERE conversation_id = ?",
                "DELETE FROM conversation_symbols WHERE conversation_id = ?",
                "DELETE FROM turn_checkpoints WHERE conversation_id = ?",
                "DELETE FROM conversations WHERE conversation_id = ?",
            ):
                connection.execute(statement, (conversation_id,))
