"""SQLite-backed store for named provider configurations.

Every SQL statement is written as an inline literal with bound ``?`` parameters:
no statement is assembled from variables, concatenation or interpolation.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from finharness.config.crypto import SecretCipher


class ConfigStoreError(RuntimeError):
    """Base class for provider configuration storage failures."""


class ConfigNotFound(ConfigStoreError):
    """Raised when a configuration id does not exist."""


class DuplicateConfigName(ConfigStoreError):
    """Raised when a configuration name is already taken."""


class ActiveConfigDeleteError(ConfigStoreError):
    """Raised when deleting the active configuration would leave another orphaned."""


@dataclass(frozen=True, slots=True)
class ProviderConfigRecord:
    id: int
    name: str
    kind: str
    base_url: str | None
    model: str
    env_key: str | None
    is_active: bool
    has_key: bool
    created_at: str
    updated_at: str


class ConfigStore:
    """Single-file SQLite store; every statement uses bound parameters."""

    def __init__(self, db_path: str | Path, *, cipher: SecretCipher) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = cipher
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS provider_configs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, kind TEXT NOT NULL, base_url TEXT, model TEXT NOT NULL, env_key TEXT, api_key_ciphertext BLOB, is_active INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_provider_configs_single_active ON provider_configs(is_active) WHERE is_active = 1")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ProviderConfigRecord:
        return ProviderConfigRecord(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            base_url=row["base_url"],
            model=row["model"],
            env_key=row["env_key"],
            is_active=bool(row["is_active"]),
            has_key=row["api_key_ciphertext"] is not None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_configs(self) -> list[ProviderConfigRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, name, kind, base_url, model, env_key, api_key_ciphertext, is_active, created_at, updated_at FROM provider_configs ORDER BY id").fetchall()
        return [self._to_record(row) for row in rows]

    def get(self, config_id: int) -> ProviderConfigRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT id, name, kind, base_url, model, env_key, api_key_ciphertext, is_active, created_at, updated_at FROM provider_configs WHERE id = ?", (config_id,)).fetchone()
        return self._to_record(row) if row is not None else None

    def get_active(self) -> ProviderConfigRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT id, name, kind, base_url, model, env_key, api_key_ciphertext, is_active, created_at, updated_at FROM provider_configs WHERE is_active = 1 LIMIT 1").fetchone()
        return self._to_record(row) if row is not None else None

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM provider_configs").fetchone()
        return int(row["n"])

    def create(
        self,
        *,
        name: str,
        kind: str,
        base_url: str | None,
        model: str,
        env_key: str | None,
        api_key: str | None,
        activate: bool,
    ) -> ProviderConfigRecord:
        now = self._now()
        ciphertext = self._cipher.encrypt(api_key) if api_key else None
        try:
            with self._connect() as connection:
                if activate:
                    connection.execute("UPDATE provider_configs SET is_active = 0 WHERE is_active = 1")
                cursor = connection.execute(
                    "INSERT INTO provider_configs (name, kind, base_url, model, env_key, api_key_ciphertext, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (name, kind, base_url, model, env_key, ciphertext, 1 if activate else 0, now, now),
                )
                config_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise DuplicateConfigName(f"配置名称已存在: {name}") from exc
        record = self.get(config_id)
        assert record is not None
        return record

    def update(
        self,
        config_id: int,
        *,
        name: str,
        kind: str,
        base_url: str | None,
        model: str,
        env_key: str | None,
        api_key: str | None,
    ) -> ProviderConfigRecord:
        """Replace every field; ``api_key=None`` keeps the stored secret."""
        if self.get(config_id) is None:
            raise ConfigNotFound(f"配置不存在: {config_id}")
        now = self._now()
        try:
            with self._connect() as connection:
                if api_key is None:
                    connection.execute(
                        "UPDATE provider_configs SET name = ?, kind = ?, base_url = ?, model = ?, env_key = ?, updated_at = ? WHERE id = ?",
                        (name, kind, base_url, model, env_key, now, config_id),
                    )
                else:
                    connection.execute(
                        "UPDATE provider_configs SET name = ?, kind = ?, base_url = ?, model = ?, env_key = ?, api_key_ciphertext = ?, updated_at = ? WHERE id = ?",
                        (name, kind, base_url, model, env_key, self._cipher.encrypt(api_key), now, config_id),
                    )
        except sqlite3.IntegrityError as exc:
            raise DuplicateConfigName(f"配置名称已存在: {name}") from exc
        record = self.get(config_id)
        assert record is not None
        return record

    def activate(self, config_id: int) -> ProviderConfigRecord:
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM provider_configs WHERE id = ?", (config_id,)).fetchone()
            if exists is None:
                raise ConfigNotFound(f"配置不存在: {config_id}")
            connection.execute("UPDATE provider_configs SET is_active = 0 WHERE is_active = 1")
            connection.execute("UPDATE provider_configs SET is_active = 1, updated_at = ? WHERE id = ?", (self._now(), config_id))
        record = self.get(config_id)
        assert record is not None
        return record

    def delete(self, config_id: int) -> None:
        record = self.get(config_id)
        if record is None:
            raise ConfigNotFound(f"配置不存在: {config_id}")
        if record.is_active and self.count() > 1:
            raise ActiveConfigDeleteError("请先切换到其它配置，再删除当前激活配置")
        with self._connect() as connection:
            connection.execute("DELETE FROM provider_configs WHERE id = ?", (config_id,))

    def resolve_key(self, config_id: int) -> str | None:
        """Return the effective secret: stored key first, then the env fallback."""
        with self._connect() as connection:
            row = connection.execute("SELECT api_key_ciphertext, env_key FROM provider_configs WHERE id = ?", (config_id,)).fetchone()
        if row is None:
            raise ConfigNotFound(f"配置不存在: {config_id}")
        if row["api_key_ciphertext"] is not None:
            return self._cipher.decrypt(row["api_key_ciphertext"])
        env_key = row["env_key"]
        if env_key:
            return os.getenv(env_key)
        return None
