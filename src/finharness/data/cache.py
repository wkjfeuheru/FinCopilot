"""LocalCache：SQLite 索引 + parquet 载荷，基于 TTL 复用。

布局：``data_cache/index.db`` 与 ``data_cache/parquet/<YYYY-MM>/<key>.parquet``。
索引仅用于定位载荷；数据本身从不存放在 SQLite 中。

每行跟踪两个摘要：

* ``lookup_key`` —— 仅由请求形状（kind + params）派生。
  检索时以此匹配，因为在实际抓取发生之前，
  数据日期是未知的。
* ``cache_key``  —— 文档规定的内容键 ``sha256(endpoint|params|date)``，
  用作 parquet 文件名，使相同内容映射到同一份载荷。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from finharness.utils.jsonx import stable_dumps
from finharness.utils.sqlite import SqliteStore


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
    """``sha256(endpoint|normalized-params|data_date)[:16]``（docs 3.5.3）。"""
    normalized = stable_dumps(params)
    digest = hashlib.sha256(f"{endpoint}|{normalized}|{data_date}".encode()).hexdigest()
    return digest[:16]


def make_lookup_key(*, kind: str, params: dict[str, Any]) -> str:
    """与日期无关的请求形状摘要，用于检索。"""
    normalized = stable_dumps(params)
    digest = hashlib.sha256(f"{kind}|{normalized}".encode()).hexdigest()
    return digest[:16]


class LocalCache(SqliteStore):
    """进程内缓存门面。所有语句均使用绑定参数。"""

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.parquet_root = self.root / "parquet"
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        super().__init__(self.root / "index.db", check_same_thread=False)
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

    # -- 读取 -----------------------------------------------------------------
    def get(self, lookup_key: str) -> tuple[pd.DataFrame, CacheEntry] | None:
        """若载荷存在且在 TTL 内则返回，否则返回 ``None``。"""
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
        """判断缓存条目是否已超过其 TTL（时间戳无法解析时视为已过期）。"""
        try:
            created = datetime.fromisoformat(entry.created_ts)
        except ValueError:
            return True
        return datetime.now().astimezone() - created > timedelta(days=entry.ttl_days)

    # -- 写入 ----------------------------------------------------------------
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
        """持久化一份载荷；空数据框不写入（docs 4.2）。"""
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
        """同步写入 parquet 文件并更新 SQLite 索引（在写入锁内于线程中调用）。"""
        cache_key = make_cache_key(endpoint=endpoint, params=params, data_date=data_date)
        month_dir = self.parquet_root / data_date[:7]
        month_dir.mkdir(parents=True, exist_ok=True)
        file_path = month_dir / f"{cache_key}.parquet"
        df.to_parquet(file_path, engine="pyarrow", index=False)
        created = datetime.now().astimezone().isoformat(timespec="seconds")
        params_json = stable_dumps(params)
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

    # -- 维护 ----------------------------------------------------------
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

    @staticmethod
    def today() -> str:
        return date.today().isoformat()
