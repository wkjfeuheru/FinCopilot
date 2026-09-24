"""记忆层持久化 DDL 与记录类型（不含存储实现）。

``store.py`` 的 §3.6.4 表结构、索引与行记录类型集中在这里：它们不持有连接、不含
行为，只是 ``MemoryStore`` 读写的数据形状。``MemoryStore`` 本身（方法共享短连接
与迁移状态）保留在 ``store.py``，两者通过本模块的常量与类型协作。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from finharness.types import Msg, ToolUse

SCHEMA_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    UNIQUE (user_id, conversation_id)
)
"""

SCHEMA_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    payload_json TEXT,
    ts TEXT NOT NULL,
    UNIQUE(user_id, conversation_id, seq),
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations(user_id, conversation_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
)
"""

SCHEMA_SUMMARY_SEGMENTS = """
CREATE TABLE IF NOT EXISTS summary_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    seq_from INTEGER NOT NULL,
    seq_to INTEGER NOT NULL,
    tier INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    ledger_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations(user_id, conversation_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
)
"""

SCHEMA_CITATIONS = """
CREATE TABLE IF NOT EXISTS citations (
    user_id TEXT NOT NULL DEFAULT '',
    cid TEXT NOT NULL,
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
    ts TEXT,
    PRIMARY KEY (user_id, conversation_id, cid)
    ,FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations(user_id, conversation_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
)
"""

SCHEMA_CONCLUSIONS = """
CREATE TABLE IF NOT EXISTS conclusions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    text TEXT NOT NULL,
    cids_json TEXT,
    ts TEXT NOT NULL,
    UNIQUE(user_id, conversation_id, subject, text),
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations(user_id, conversation_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
)
"""

SCHEMA_SYMBOLS = """
CREATE TABLE IF NOT EXISTS conversation_symbols (
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    PRIMARY KEY (user_id, conversation_id, symbol),
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations(user_id, conversation_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
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
    indexed_at TEXT,
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
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    rounds INTEGER NOT NULL DEFAULT 0,
    turn_index INTEGER NOT NULL DEFAULT 0,
    plan_json TEXT,
    partial_answer TEXT,
    persisted_seq INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, conversation_id),
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations(user_id, conversation_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
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
