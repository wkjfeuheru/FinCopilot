"""基于 SQLite 的具名 Provider 配置存储（按用户隔离）。

每条 SQL 语句都以字面量内联书写并使用 ``?`` 绑定参数：
任何语句都不由变量拼接、字符串连接或插值组装而成。

作用域：每条配置归属一个 ``user_id``。名称唯一与“单一激活”约束
都按用户计算——config_id 是全局自增句柄，因此每个按 id 的方法
都必须带归属校验，否则一个用户可以改名、激活或删除另一个用户的配置。
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from finharness.config.crypto import SecretCipher
from finharness.utils.clock import utc_now_iso
from finharness.utils.sqlite import SqliteStore


class ConfigStoreError(RuntimeError):
    """Provider 配置存储失败的基类。"""


class ConfigNotFound(ConfigStoreError):
    """配置 id 不存在时抛出。"""


class DuplicateConfigName(ConfigStoreError):
    """配置名称已被占用时抛出。"""


class ActiveConfigDeleteError(ConfigStoreError):
    """删除激活配置会导致其它配置失去归属时抛出。"""


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
    user_id: str = ""


_CONFIG_COLUMNS = (
    "id, user_id, name, kind, base_url, model, env_key, api_key_ciphertext,"
    " is_active, created_at, updated_at"
)

SCHEMA_PROVIDER_CONFIGS_V2 = """
CREATE TABLE provider_configs_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    base_url TEXT,
    model TEXT NOT NULL,
    env_key TEXT,
    api_key_ciphertext BLOB,
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id, name)
)
"""

INDEX_SINGLE_ACTIVE_PER_USER = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_provider_configs_single_active"
    " ON provider_configs(user_id) WHERE is_active = 1"
)


class ConfigStore(SqliteStore):
    """单文件 SQLite 存储；每条语句都使用绑定参数。"""

    def __init__(self, db_path: str | Path, *, cipher: SecretCipher) -> None:
        """打开/创建数据库并确保表结构就绪；``cipher`` 用于加解密 api_key。"""

        super().__init__(db_path)
        self._cipher = cipher
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            self._migrate(connection)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """建表或把 v1（无 user_id、全局唯一名）原地升级为按用户的 v2。

        v1 表携带 ``UNIQUE(name)`` 与全局单激活部分索引，二者都必须
        改为按用户，因此迁移走建新表-拷贝-替换，而非 ``ALTER TABLE``。
        """
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(provider_configs)")
        }
        if not columns:
            connection.execute(SCHEMA_PROVIDER_CONFIGS_V2)
            connection.execute(
                "ALTER TABLE provider_configs_v2 RENAME TO provider_configs"
            )
        elif "user_id" not in columns:
            connection.execute("DROP INDEX IF EXISTS idx_provider_configs_single_active")
            connection.execute(SCHEMA_PROVIDER_CONFIGS_V2)
            connection.execute(
                "INSERT INTO provider_configs_v2 (id, user_id, name, kind, base_url,"
                " model, env_key, api_key_ciphertext, is_active, created_at, updated_at)"
                " SELECT id, '', name, kind, base_url, model, env_key,"
                " api_key_ciphertext, is_active, created_at, updated_at"
                " FROM provider_configs"
            )
            connection.execute("DROP TABLE provider_configs")
            connection.execute(
                "ALTER TABLE provider_configs_v2 RENAME TO provider_configs"
            )
        connection.execute(INDEX_SINGLE_ACTIVE_PER_USER)

    def claim_user(self, user_id: str) -> int:
        """把无主（``user_id=''``）的配置划归指定用户；返回认领条数。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE provider_configs SET user_id = ? WHERE user_id = ''", (user_id,)
            )
        return cursor.rowcount

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ProviderConfigRecord:
        return ProviderConfigRecord(
            id=row["id"],
            user_id=row["user_id"],
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

    def list_configs(self, *, user_id: str = "") -> list[ProviderConfigRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {_CONFIG_COLUMNS} FROM provider_configs WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def get(self, config_id: int, *, user_id: str = "") -> ProviderConfigRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_CONFIG_COLUMNS} FROM provider_configs WHERE id = ? AND user_id = ?",
                (config_id, user_id),
            ).fetchone()
        return self._to_record(row) if row is not None else None

    def get_active(self, *, user_id: str = "") -> ProviderConfigRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_CONFIG_COLUMNS} FROM provider_configs"
                " WHERE user_id = ? AND is_active = 1 LIMIT 1",
                (user_id,),
            ).fetchone()
        return self._to_record(row) if row is not None else None

    def count(self, *, user_id: str = "") -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM provider_configs WHERE user_id = ?", (user_id,)
            ).fetchone()
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
        user_id: str = "",
    ) -> ProviderConfigRecord:
        """插入一条配置；``activate`` 为 True 时将其设为该用户的唯一激活配置。

        api_key 非空时加密存储。名称重复抛出 ``DuplicateConfigName``。
        返回新创建记录的不可变快照。
        """

        now = utc_now_iso()
        ciphertext = self._cipher.encrypt(api_key) if api_key else None
        try:
            with self._connect() as connection:
                if activate:
                    connection.execute(
                        "UPDATE provider_configs SET is_active = 0 WHERE user_id = ? AND is_active = 1",
                        (user_id,),
                    )
                cursor = connection.execute(
                    "INSERT INTO provider_configs (user_id, name, kind, base_url, model, env_key, api_key_ciphertext, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, name, kind, base_url, model, env_key, ciphertext, 1 if activate else 0, now, now),
                )
                config_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise DuplicateConfigName(f"配置名称已存在: {name}") from exc
        record = self.get(config_id, user_id=user_id)
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
        user_id: str = "",
    ) -> ProviderConfigRecord:
        """替换记录的每个字段；``api_key=None`` 时保留已存储的密钥。

        配置不存在抛出 ``ConfigNotFound``，名称重复抛出 ``DuplicateConfigName``。
        返回更新后的记录快照。
        """

        if self.get(config_id, user_id=user_id) is None:
            raise ConfigNotFound(f"配置不存在: {config_id}")
        now = utc_now_iso()
        try:
            with self._connect() as connection:
                if api_key is None:
                    connection.execute(
                        "UPDATE provider_configs SET name = ?, kind = ?, base_url = ?, model = ?, env_key = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                        (name, kind, base_url, model, env_key, now, config_id, user_id),
                    )
                else:
                    connection.execute(
                        "UPDATE provider_configs SET name = ?, kind = ?, base_url = ?, model = ?, env_key = ?, api_key_ciphertext = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                        (name, kind, base_url, model, env_key, self._cipher.encrypt(api_key), now, config_id, user_id),
                    )
        except sqlite3.IntegrityError as exc:
            raise DuplicateConfigName(f"配置名称已存在: {name}") from exc
        record = self.get(config_id, user_id=user_id)
        assert record is not None
        return record

    def activate(self, config_id: int, *, user_id: str = "") -> ProviderConfigRecord:
        """将指定配置设为该用户的唯一激活项；不存在时抛出 ``ConfigNotFound``。"""

        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM provider_configs WHERE id = ? AND user_id = ?",
                (config_id, user_id),
            ).fetchone()
            if exists is None:
                raise ConfigNotFound(f"配置不存在: {config_id}")
            connection.execute(
                "UPDATE provider_configs SET is_active = 0 WHERE user_id = ? AND is_active = 1",
                (user_id,),
            )
            connection.execute(
                "UPDATE provider_configs SET is_active = 1, updated_at = ? WHERE id = ? AND user_id = ?",
                (utc_now_iso(), config_id, user_id),
            )
        record = self.get(config_id, user_id=user_id)
        assert record is not None
        return record

    def delete(self, config_id: int, *, user_id: str = "") -> None:
        """删除配置。

        不存在时抛出 ``ConfigNotFound``；若删除的是激活配置且该用户尚存
        其它配置，则抛出 ``ActiveConfigDeleteError``（需先切换激活项）。
        """

        record = self.get(config_id, user_id=user_id)
        if record is None:
            raise ConfigNotFound(f"配置不存在: {config_id}")
        if record.is_active and self.count(user_id=user_id) > 1:
            raise ActiveConfigDeleteError("请先切换到其它配置，再删除当前激活配置")
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM provider_configs WHERE id = ? AND user_id = ?",
                (config_id, user_id),
            )

    def resolve_key(self, config_id: int, *, user_id: str = "") -> str | None:
        """返回生效的密钥：优先使用已存储的密钥，否则回退到环境变量。

        配置不存在抛出 ``ConfigNotFound``；两者都无则返回 None。
        """

        with self._connect() as connection:
            row = connection.execute(
                "SELECT api_key_ciphertext, env_key FROM provider_configs WHERE id = ? AND user_id = ?",
                (config_id, user_id),
            ).fetchone()
        if row is None:
            raise ConfigNotFound(f"配置不存在: {config_id}")
        if row["api_key_ciphertext"] is not None:
            return self._cipher.decrypt(row["api_key_ciphertext"])
        env_key = row["env_key"]
        if env_key:
            return os.getenv(env_key)
        return None
