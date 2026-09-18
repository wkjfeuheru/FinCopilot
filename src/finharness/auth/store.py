"""用户与会话令牌存储（SQLite，users.db）。

* ``users``：用户名唯一（大小写不敏感）、密码只存 PBKDF2 哈希。
* ``auth_sessions``：不透明令牌的会话表。令牌本体（``t_`` + 32 字节随机）
  只发给客户端一次；库中只存其 SHA-256，因此库文件泄露也无法冒充登录。

迁移规则：存量单用户安装没有用户概念。第一个注册的账号会认领
``user_id=''`` 的全部遗产（对话、笔记、供应商配置），升级后数据不丢。
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from finharness.auth.passwords import hash_password, verify_password

SCHEMA_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

SCHEMA_AUTH_SESSIONS = """
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)
"""

INDEX_AUTH_SESSIONS_USER = (
    "CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id)"
)

# 用户名：字母/数字/下划线/连字符/中文，2-32 字符（长度另由 register 校验）。
_USERNAME_RE = re.compile(r"^[\w\u4e00-\u9fff-]+$", re.UNICODE)

_USERNAME_MIN = 2
_USERNAME_MAX = 32
_PASSWORD_MIN = 8


class UserStoreError(RuntimeError):
    """用户存储操作失败的基类。"""


class DuplicateUsername(UserStoreError):
    """用户名已被占用时抛出。"""


class UserNotFound(UserStoreError):
    """用户不存在时抛出。"""


class InvalidCredentials(UserStoreError):
    """登录凭据不正确时抛出（用户不存在与密码错误同消息，避免枚举）。"""


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """认证通过后贯穿整个请求的用户身份。"""

    id: str
    username: str


@dataclass(frozen=True, slots=True)
class IssuedSession:
    """登录/注册成功后签发的会话：令牌只在此处出现一次。"""

    token: str
    user: CurrentUser
    expires_at: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_user_id() -> str:
    return f"u_{secrets.token_hex(8)}"


def _validate_credentials(username: str, password: str, *, min_password_len: int) -> None:
    """注册/登录前的本地校验；不通过抛出 ``UserStoreError``。"""
    if not _USERNAME_MIN <= len(username) <= _USERNAME_MAX:
        raise UserStoreError(f"用户名长度须在 {_USERNAME_MIN}-{_USERNAME_MAX} 个字符之间")
    if not _USERNAME_RE.fullmatch(username):
        raise UserStoreError("用户名只能包含字母、数字、下划线、连字符或中文")
    if len(password) < max(min_password_len, _PASSWORD_MIN):
        raise UserStoreError(f"密码至少需要 {max(min_password_len, _PASSWORD_MIN)} 个字符")


class UserStore:
    """基于 SQLite 的用户与会话令牌存储；所有语句都使用绑定参数。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(SCHEMA_USERS)
            connection.execute(SCHEMA_AUTH_SESSIONS)
            connection.execute(INDEX_AUTH_SESSIONS_USER)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    # -- 用户 ------------------------------------------------------------------
    def count_users(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])

    def get_user(self, user_id: str) -> CurrentUser | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return CurrentUser(id=row["id"], username=row["username"]) if row else None

    def find_by_username(self, username: str) -> CurrentUser | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()
        return CurrentUser(id=row["id"], username=row["username"]) if row else None

    def register(
        self,
        username: str,
        password: str,
        *,
        min_password_len: int = 8,
        ttl_s: int = 14 * 24 * 3600,
        claim_legacy: "callable | None" = None,
    ) -> IssuedSession:
        """注册新用户并立即签发会话。

        ``claim_legacy`` 在 users 表原本为空时被调用（传入新 user id），
        用于把单用户时代的存量数据划归第一个注册的用户。
        ``DuplicateUsername`` / ``UserStoreError`` 由调用方映射为 HTTP 状态。
        """
        _validate_credentials(username, password, min_password_len=min_password_len)
        now = _now()
        user_id = _new_user_id()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
            if existing is not None:
                raise DuplicateUsername(f"用户名已存在: {username}")
            was_empty = connection.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] == 0
            connection.execute(
                "INSERT INTO users (id, username, password_hash, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (user_id, username, hash_password(password), now.isoformat(), now.isoformat()),
            )
        if was_empty and claim_legacy is not None:
            claim_legacy(user_id)
        return self._issue(user_id, ttl_s=ttl_s)

    def login(
        self,
        username: str,
        password: str,
        *,
        min_password_len: int = 8,
        ttl_s: int = 14 * 24 * 3600,
    ) -> IssuedSession:
        """校验凭据并签发会话；失败统一为 ``InvalidCredentials``。"""
        _validate_credentials(username, password, min_password_len=min_password_len)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()
        # 用户不存在也做一次哈希校验，使两种失败路径耗时接近，支持时间上不可区分。
        stored = row["password_hash"] if row is not None else hash_password("timing-padding")
        if row is None or not verify_password(password, stored):
            raise InvalidCredentials("用户名或密码不正确")
        return self._issue(row["id"], ttl_s=ttl_s)

    # -- 会话令牌 --------------------------------------------------------------
    def _issue(self, user_id: str, *, ttl_s: int) -> IssuedSession:
        """生成令牌、落库哈希并清理过期行。"""
        token = f"t_{secrets.token_hex(32)}"
        now = _now()
        expires = now + timedelta(seconds=ttl_s)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (_token_hash(token), user_id, now.isoformat(), expires.isoformat()),
            )
            connection.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now.isoformat(),))
        user = self.get_user(user_id)
        assert user is not None
        return IssuedSession(token=token, user=user, expires_at=expires.isoformat())

    def resolve_token(self, token: str) -> CurrentUser | None:
        """令牌有效时返回所属用户；未知/过期/已撤销一律 None。"""
        if not token:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT s.user_id, u.username FROM auth_sessions s"
                " JOIN users u ON u.id = s.user_id"
                " WHERE s.token_hash = ? AND s.expires_at >= ?",
                (_token_hash(token), _now().isoformat()),
            ).fetchone()
        return CurrentUser(id=row["user_id"], username=row["username"]) if row else None

    def revoke(self, token: str) -> bool:
        """撤销一个会话令牌；未知令牌返回 False。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),)
            )
        return cursor.rowcount > 0
