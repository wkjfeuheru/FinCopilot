"""LocalCache: SQLite index + parquet payloads with TTL-based reuse.

Layout: ``data_cache/index.db`` and ``data_cache/parquet/<YYYY-MM>/<key>.parquet``.
The index only locates payloads; the data itself never lives in SQLite.

Two digests are tracked per row:

* ``lookup_key`` — derived from the request shape (kind + params) only. This is
  what retrieval matches on, because the data date is unknown until a fetch
  happens.
* ``cache_key``  — the documented content key ``sha256(endpoint|params|date)``,
  used as the parquet filename so identical content maps to one payload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class CacheStats:
    entries: int
    hits: int
    misses: int
    hit_ratio: float


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: str
    lookup_key: str
    endpoint: str
    params_json: str
    data_date: str
    rows: int
    file_path: str
    ttl_days: int
    created_ts: str


def make_cache_key(*, endpoint: str, params: dict[str, Any], data_date: str) -> str:
    """``sha256(endpoint|normalized-params|data_date)[:16]`` (docs 3.5.3)."""
    normalized = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{endpoint}|{normalized}|{data_date}".encode("utf-8")).hexdigest()
    return digest[:16]


def make_lookup_key(*, kind: str, params: dict[str, Any]) -> str:
    """Date-independent digest of the request shape, used for retrieval."""
    normalized = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{kind}|{normalized}".encode("utf-8")).hexdigest()
    return digest[:16]


class LocalCache:
    """Process-local cache facade. All statements use bound parameters."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.parquet_root = self.root / "parquet"
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        self._hits = 0
        self._misses = 0
        self._write_lock = asyncio.Lock()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cache_index (cache_key TEXT PRIMARY KEY, lookup_key TEXT NOT NULL, endpoint TEXT NOT NULL, params_json TEXT NOT NULL, data_date TEXT NOT NULL, rows INTEGER NOT NULL, cols_json TEXT NOT NULL, created_ts TEXT NOT NULL, ttl_days INTEGER NOT NULL, file_path TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_cache_lookup ON cache_index(lookup_key)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_cache_endpoint ON cache_index(endpoint)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_cache_data_date ON cache_index(data_date)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    # -- read -----------------------------------------------------------------
    def get(self, lookup_key: str) -> tuple[pd.DataFrame, CacheEntry] | None:
        """Return a payload when present and within TTL, else ``None``."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cache_key, lookup_key, endpoint, params_json, data_date, rows, file_path, ttl_days, created_ts FROM cache_index WHERE lookup_key = ?",
                (lookup_key,),
            ).fetchone()
        if row is None:
            self._misses += 1
            return None
        entry = CacheEntry(
            cache_key=row["cache_key"],
            lookup_key=row["lookup_key"],
            endpoint=row["endpoint"],
            params_json=row["params_json"],
            data_date=row["data_date"],
            rows=int(row["rows"]),
            file_path=row["file_path"],
            ttl_days=int(row["ttl_days"]),
            created_ts=row["created_ts"],
        )
        if self._is_expired(entry):
            self._misses += 1
            return None
        path = Path(entry.file_path)
        if not path.is_file():
            self._misses += 1
            return None
        self._hits += 1
        return pd.read_parquet(path), entry

    @staticmethod
    def _is_expired(entry: CacheEntry) -> bool:
        try:
            created = datetime.fromisoformat(entry.created_ts)
        except ValueError:
            return True
        return datetime.now().astimezone() - created > timedelta(days=entry.ttl_days)

    # -- write ----------------------------------------------------------------
    async def put(
        self,
        *,
        lookup_key: str,
        endpoint: str,
        params: dict[str, Any],
        data_date: str,
        df: pd.DataFrame | None,
        ttl_days: int,
    ) -> CacheEntry | None:
        """Persist a payload; empty frames are not written (docs 4.2)."""
        if df is None or len(df) == 0:
            return None
        async with self._write_lock:
            return await asyncio.to_thread(
                self._write_sync,
                lookup_key=lookup_key,
                endpoint=endpoint,
                params=params,
                data_date=data_date,
                df=df,
                ttl_days=ttl_days,
            )

    def _write_sync(
        self,
        *,
        lookup_key: str,
        endpoint: str,
        params: dict[str, Any],
        data_date: str,
        df: pd.DataFrame,
        ttl_days: int,
    ) -> CacheEntry:
        cache_key = make_cache_key(endpoint=endpoint, params=params, data_date=data_date)
        month_dir = self.parquet_root / data_date[:7]
        month_dir.mkdir(parents=True, exist_ok=True)
        file_path = month_dir / f"{cache_key}.parquet"
        df.to_parquet(file_path, engine="pyarrow", index=False)
        created = datetime.now().astimezone().isoformat(timespec="seconds")
        params_json = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
        cols_json = json.dumps([str(c) for c in df.columns], ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO cache_index (cache_key, lookup_key, endpoint, params_json, data_date, rows, cols_json, created_ts, ttl_days, file_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cache_key, lookup_key, endpoint, params_json, data_date, int(len(df)), cols_json, created, int(ttl_days), str(file_path)),
            )
        return CacheEntry(
            cache_key=cache_key, lookup_key=lookup_key, endpoint=endpoint,
            params_json=params_json, data_date=data_date, rows=int(len(df)),
            file_path=str(file_path), ttl_days=int(ttl_days), created_ts=created,
        )

    # -- maintenance ----------------------------------------------------------
    def stats(self) -> CacheStats:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM cache_index").fetchone()
        entries = int(row["n"])
        total = self._hits + self._misses
        return CacheStats(
            entries=entries,
            hits=self._hits,
            misses=self._misses,
            hit_ratio=(self._hits / total) if total else 0.0,
        )

    def gc(self, *, min_entries: int = 1000) -> int:
        """Drop expired rows and their payloads; skipped when the cache is small."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT cache_key, file_path, ttl_days, created_ts FROM cache_index"
            ).fetchall()
            if len(rows) < min_entries:
                return 0
            removed = 0
            for row in rows:
                entry = CacheEntry(
                    cache_key=row["cache_key"], lookup_key="", endpoint="", params_json="",
                    data_date="", rows=0, file_path=row["file_path"],
                    ttl_days=int(row["ttl_days"]), created_ts=row["created_ts"],
                )
                if not self._is_expired(entry):
                    continue
                connection.execute(
                    "DELETE FROM cache_index WHERE cache_key = ?", (row["cache_key"],)
                )
                Path(row["file_path"]).unlink(missing_ok=True)
                removed += 1
        return removed

    @staticmethod
    def today() -> str:
        return date.today().isoformat()
