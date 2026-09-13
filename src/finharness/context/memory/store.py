"""Conversation memory store: the persistent half of the memory layers.

One SQLite file (``settings.paths.memory_db``) holds the conversation transcript
(retrievable by id) plus the structured memory built on top of it: summary
segments, citations, conclusions and per-conversation symbol pools.

Scope differs by table and that is deliberate:

* everything keyed by ``conversation_id`` is isolated — one conversation cannot
  see another's transcript, summaries, citations or conclusions;
* ``notes`` carries no conversation scope: user preferences are shared by every
  conversation, because "how this user likes reports" is not conversation-local.

Every statement uses bound parameters; none is assembled from variables.
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
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'preference',
    updated_at TEXT NOT NULL
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


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation_id: str
    title: str | None
    created_at: str
    updated_at: str
    last_active_at: str


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
    """SQLite-backed conversation memory; all statements use bound parameters."""

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
            connection.execute(INDEX_MESSAGES_BY_CONVERSATION)
            connection.execute(INDEX_CONCLUSIONS_BY_SUBJECT)
            connection.execute(INDEX_CITATIONS_BY_CONVERSATION)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    # -- conversations --------------------------------------------------------
    def ensure_conversation(self, conversation_id: str, *, title: str | None = None) -> ConversationRecord:
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id, title, created_at, updated_at, last_active_at FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO conversations (conversation_id, title, created_at, updated_at, last_active_at) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, title, now, now, now),
                )
            else:
                # A later title only fills a blank one; it never overwrites.
                connection.execute(
                    "UPDATE conversations SET updated_at = ?, last_active_at = ?, title = COALESCE(title, ?) WHERE conversation_id = ?",
                    (now, now, title, conversation_id),
                )
        record = self.get_conversation(conversation_id)
        assert record is not None
        return record

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id, title, created_at, updated_at, last_active_at FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return ConversationRecord(
            conversation_id=row["conversation_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_active_at=row["last_active_at"],
        )

    def list_conversations(self, *, limit: int = 50) -> list[ConversationRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT conversation_id, title, created_at, updated_at, last_active_at FROM conversations ORDER BY last_active_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            ConversationRecord(
                conversation_id=row["conversation_id"],
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

    # -- transcript -----------------------------------------------------------
    def append_messages(self, conversation_id: str, messages: list[Msg]) -> tuple[int, int]:
        """Write messages in one transaction; returns the assigned seq range."""
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
        """Attach replay-only metadata to the newest persisted final answer."""
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
        """Retrieve one message by its row id (Q8: internal retrieval by id)."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT role, content, payload_json FROM messages WHERE id = ?",
                (int(message_id),),
            ).fetchone()
        return _decode_message(row) if row is not None else None

    # -- summary segments -----------------------------------------------------
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
        """Rewrite the segment set for a conversation, preserving seq ranges."""
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

    # -- citations (cid must survive a restart verbatim) ----------------------
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

    # -- conclusions ----------------------------------------------------------
    def save_conclusion(
        self, conversation_id: str, *, subject: str, text: str, cids: list[str] | None = None
    ) -> None:
        """Idempotent: the same conclusion under the same subject is stored once."""
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
        """Recall by subject keys; one statement per key keeps SQL static."""
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

    # -- symbol pool ----------------------------------------------------------
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

    # -- preferences (global by design: shared across conversations) ----------
    def set_note(self, key: str, value: str, *, kind: str = "preference") -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO notes (key, value, kind, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, kind = excluded.kind, updated_at = excluded.updated_at",
                (key, value, kind, _now()),
            )

    def get_notes(self) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM notes ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    # -- retention ------------------------------------------------------------
    def prune(self, *, max_conversations: int, max_age_days: int) -> list[str]:
        """Drop conversations past either bound, with their scoped memory."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat(timespec="seconds")
        with self._connect() as connection:
            stale_rows = connection.execute(
                "SELECT conversation_id FROM conversations WHERE last_active_at < ?",
                (cutoff,),
            ).fetchall()
            excess_rows = connection.execute(
                "SELECT conversation_id FROM conversations ORDER BY last_active_at DESC LIMIT -1 OFFSET ?",
                (int(max_conversations),),
            ).fetchall()
        doomed = {row["conversation_id"] for row in stale_rows}
        doomed.update(row["conversation_id"] for row in excess_rows)
        for conversation_id in doomed:
            self.delete_conversation(conversation_id)
        return sorted(doomed)

    def delete_conversation(self, conversation_id: str) -> None:
        """Remove a conversation and everything scoped to it (notebook excluded)."""
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
