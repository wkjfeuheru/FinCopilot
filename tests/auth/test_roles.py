"""角色与鉴权测试：role 列迁移、admin bootstrap、require_admin 依赖。

背景（docs 03.13 隔离方案）：users 表原本没有角色概念，唯一的管理员机制是
``trace_store.admin_users`` 用户名白名单。本模块锁定新契约：

* 存量库自动补 ``role`` 列（默认 'user'），旧数据零操作升级；
* ``auth.admin_bootstrap=true`` 期间注册的用户 role='admin'，关闭后注册的
  是普通用户——这是运维在公网上安全地产生第一个管理员的开关；
* ``create_require_admin`` 依赖：未登录 401、已登录非管理员 403。
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from finharness.auth.dependency import create_require_admin, create_require_user
from finharness.auth.store import CurrentUser, UserStore
from finharness.config.settings import Settings
from finharness.server.api import create_app


def _store(tmp_path, *, settings: Settings | None = None) -> UserStore:
    paths = settings.paths if settings else None
    db = (paths.auth_db if paths else tmp_path / "state" / "users.db")
    return UserStore(db)


def test_new_users_database_has_role_column(tmp_path) -> None:
    store = _store(tmp_path)
    session = store.register("alice", "password-123")
    assert session.user.role == "user"

    with sqlite3.connect(store.db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(users)")}
    assert "role" in columns


def test_legacy_database_is_migrated_with_default_role(tmp_path) -> None:
    """旧库（无 role 列）打开后自动补列，存量用户一律 user。"""
    db_path = tmp_path / "state" / "users.db"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE users ("
            " id TEXT PRIMARY KEY,"
            " username TEXT NOT NULL UNIQUE COLLATE NOCASE,"
            " password_hash TEXT NOT NULL,"
            " created_at TEXT NOT NULL,"
            " updated_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO users (id, username, password_hash, created_at, updated_at)"
            " VALUES ('u_legacy', 'legacy', 'x', '2025-01-01', '2025-01-01')"
        )

    store = UserStore(db_path)
    user = store.get_user("u_legacy")
    assert user is not None
    assert user.role == "user"
    assert store.list_users() == [
        {"id": "u_legacy", "username": "legacy", "role": "user", "created_at": "2025-01-01"}
    ]


def test_admin_bootstrap_registers_admin_and_later_users_stay_user(tmp_path) -> None:
    """开关语义：bootstrap_admin=True 时注册为 admin；不传（关闭）为 user。"""
    store = _store(tmp_path)
    admin = store.register(
        "boss", "password-123", min_password_len=8, ttl_s=60, bootstrap_admin=True
    )
    assert admin.user.role == "admin"

    follower = store.register("alice", "password-123", min_password_len=8, ttl_s=60)
    assert follower.user.role == "user"

    resolved = store.resolve_token(admin.token)
    assert resolved is not None and resolved.role == "admin"


def test_require_admin_gates_401_and_403(tmp_path) -> None:
    store = _store(tmp_path)
    admin_session = store.register("boss", "password-123", bootstrap_admin=True)
    user_session = store.register("alice", "password-123")

    app = FastAPI()
    require_user = create_require_user(store)
    require_admin = create_require_admin(require_user)

    @app.get("/probe")
    def probe(user: CurrentUser = Depends(require_admin)):  # noqa: F821
        return {"username": user.username}

    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/probe").status_code == 401

    ok = client.get(
        "/probe", headers={"Authorization": f"Bearer {admin_session.token}"}
    )
    assert ok.status_code == 200 and ok.json() == {"username": "boss"}

    forbidden = client.get(
        "/probe", headers={"Authorization": f"Bearer {user_session.token}"}
    )
    assert forbidden.status_code == 403


def test_me_returns_role(tmp_path) -> None:
    """/v1/auth/me 携带 role，前端据此决定是否渲染管理入口。

    同时锁定 bootstrap 的运维语义：开关开启期间注册的是 admin；
    关闭后（重新构建 app，模拟 Railway 删变量重新部署）注册的是普通用户。
    """
    import json

    def payload(bootstrap: bool) -> dict:
        return {
            "auth": {"admin_bootstrap": bootstrap, "allow_register": True},
            "data": {"cache_dir": (tmp_path / "cache").as_posix()},
            "paths": {"state_dir": (tmp_path / "state").as_posix()},
        }

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(payload(True)), encoding="utf-8")
    client = TestClient(create_app(settings=Settings.from_file(settings_path)))
    register = client.post(
        "/v1/auth/register", json={"username": "boss", "password": "password-123"}
    )
    assert register.status_code == 200
    assert register.json()["user"]["role"] == "admin"
    token = register.json()["token"]

    me = client.get(
        "/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    ).json()["user"]
    assert me["role"] == "admin"

    # 运维关闭开关（重新部署）；已注册的 boss 仍是 admin（role 落库不回填）。
    settings_path.write_text(json.dumps(payload(False)), encoding="utf-8")
    closed = TestClient(create_app(settings=Settings.from_file(settings_path)))
    second = closed.post(
        "/v1/auth/register", json={"username": "alice", "password": "password-123"}
    )
    assert second.json()["user"]["role"] == "user"
    me2 = closed.get(
        "/v1/auth/me",
        headers={"Authorization": f"Bearer {second.json()['token']}"},
    ).json()["user"]
    assert me2["role"] == "user"


def test_env_override_for_admin_bootstrap(tmp_path, monkeypatch) -> None:
    """Railway 部署经环境变量开启 bootstrap（白名单变量必须放行）。"""
    monkeypatch.setenv("FINH_AUTH_ADMIN_BOOTSTRAP", "true")
    settings = Settings.from_file(tmp_path / "absent-settings.json")
    assert settings.auth.admin_bootstrap is True


@pytest.mark.parametrize("raw", ["nope", "1.5"])
def test_env_override_rejects_invalid_bool(raw, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FINH_AUTH_ADMIN_BOOTSTRAP", raw)
    with pytest.raises(Exception, match="FINH_AUTH_ADMIN_BOOTSTRAP"):
        Settings.from_file(tmp_path / "absent-settings.json")


def test_usage_ledger_records_turns_without_trace(tmp_path) -> None:
    """账本与 trace 开关解耦：监控关闭时每轮对话仍落一行用量。"""
    from finharness.observability.usage_store import UsageStore
    from finharness.provider.fake import FakeProvider

    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "state_dir": tmp_path / "state",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
            "config_db": tmp_path / "state" / "config.db",
            "secret_key": tmp_path / "state" / "secret.key",
            "usage_db": tmp_path / "state" / "usage.db",
        },
    )
    client = TestClient(create_app(FakeProvider(["ok"]), settings=settings))
    register = client.post(
        "/v1/auth/register", json={"username": "alice", "password": "password-123"}
    )
    assert register.status_code == 200, register.text
    headers = {"Authorization": f"Bearer {register.json()['token']}"}

    response = client.post(
        "/v1/chat/stream", json={"message": "hi"}, headers=headers
    )
    assert response.status_code == 200
    assert "event: done" in response.text

    ledger = UsageStore(settings.paths.usage_db)
    totals = ledger.totals_by_user()
    assert len(totals) == 1
    assert totals[0]["turns"] == 1
    summary = ledger.summary()
    assert summary["turns"] == 1
    assert summary["active_users"] == 1
