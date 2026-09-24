"""管理 API 测试：/v1/admin/* 的鉴权边界与三源聚合。

权限语义（docs 03.13 + 管理员页设计）：未登录 401、已登录非管理员 403、
管理员 200。数据合并三源——UserStore（账号）、memory.db（对话数）、
usage.db（token/轮数）。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from finharness.auth.store import UserStore
from finharness.config.settings import Settings
from finharness.context.memory.store import MemoryStore
from finharness.observability.usage_store import UsageStore
from finharness.server.api import create_app


def _settings(tmp_path) -> Settings:
    return Settings(
        auth={"admin_bootstrap": True},
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


def _register(client: TestClient, username: str) -> dict:
    response = client.post(
        "/v1/auth/register", json={"username": username, "password": "password-123"}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_admin_endpoints_gate_anonymous_and_non_admin(tmp_path) -> None:
    """鉴权三态：匿名 401、普通用户 403、管理员 200。

    普通用户由 UserStore 直接造（bootstrap 开关开着时 HTTP 注册只会产生
    管理员——关闭开关需重新部署，属于运维流程而非请求期状态）。
    Bearer 断言前清 cookie：注册响应 set-cookie 了管理员会话，而
    ``resolve_user`` 是 cookie 优先——带着它会冒充管理员身份。
    """
    settings = _settings(tmp_path)
    anon = TestClient(create_app(settings=settings))
    assert anon.get("/v1/admin/status").status_code == 401
    assert anon.get("/v1/admin/users").status_code == 401

    admin = _register(anon, "boss")
    assert admin["user"]["role"] == "admin"
    # 直接落库一个普通用户（绕过 bootstrap 开关），模拟"关闭开关后注册"。
    store = UserStore(settings.paths.auth_db)
    member = store.register("alice", "password-123")
    assert member.user.role == "user"

    anon.cookies.clear()
    member_headers = {"Authorization": f"Bearer {member.token}"}
    assert anon.get("/v1/admin/status", headers=member_headers).status_code == 403
    assert anon.get("/v1/admin/users", headers=member_headers).status_code == 403

    admin_headers = {"Authorization": f"Bearer {admin['token']}"}
    status = anon.get("/v1/admin/status", headers=admin_headers)
    assert status.status_code == 200 and status.json() == {"is_admin": True}


def test_admin_users_merges_three_sources(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings=settings))
    admin = _register(client, "boss")
    # 普通用户（bootstrap 关闭态的等价物，直接落库）
    store = UserStore(settings.paths.auth_db)
    member = store.register("alice", "password-123")
    member_id = member.user.id

    # alice 有 2 个对话、3 轮用量；boss 无数据
    memory = MemoryStore(settings.paths.memory_db)
    for name in ("conv-a", "conv-b"):
        memory.ensure_conversation(f"c_{name}", user_id=member_id, title=name)
    usage = UsageStore(settings.paths.usage_db)
    for _ in range(3):
        usage.record_turn(
            user_id=member_id,
            input_tokens=100,
            output_tokens=40,
            cache_hit_tokens=10,
            status="done",
        )

    headers = {"Authorization": f"Bearer {admin['token']}"}
    response = client.get("/v1/admin/users", headers=headers)
    assert response.status_code == 200
    rows = {row["username"]: row for row in response.json()["users"]}

    assert rows["boss"]["role"] == "admin"
    assert rows["boss"]["conversations"] == 0
    assert rows["boss"]["turns"] == 0

    assert rows["alice"]["role"] == "user"
    assert rows["alice"]["conversations"] == 2
    assert rows["alice"]["turns"] == 3
    assert rows["alice"]["input_tokens"] == 300
    assert rows["alice"]["output_tokens"] == 120
    assert rows["alice"]["cache_hit_tokens"] == 30
    assert rows["alice"]["last_active_at"]


def test_admin_users_window_filter(tmp_path) -> None:
    """窗口过滤只影响窗口列：总累计列始终全量。"""
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings=settings))
    admin = _register(client, "boss")
    usage = UsageStore(settings.paths.usage_db)
    usage.record_turn(user_id="u_1", input_tokens=5, output_tokens=5, status="done")
    with __import__("sqlite3").connect(usage.db_path) as connection:
        connection.execute(
            "INSERT INTO usage_turns (user_id, ts, input_tokens, output_tokens,"
            " cache_hit_tokens, duration_ms, status) VALUES (?, ?, 999, 999, 0, 0, 'done')",
            ("u_1", "2020-01-01T00:00:00+00:00"),
        )

    headers = {"Authorization": f"Bearer {admin['token']}"}
    body = client.get(
        "/v1/admin/users", headers=headers, params={"window": "24h"}
    ).json()
    row = next(r for r in body["users"] if r["username"] == "boss")
    # 总累计列（turns/input_tokens）不受窗口影响——但 u_1 不是注册用户，
    # 不会出现在行里；boss 自身 0 轮。窗口过滤的效果在 summary 里验证。
    assert row["turns"] == 0

    summary = client.get(
        "/v1/admin/usage/summary", headers=headers, params={"window": "24h"}
    ).json()
    assert summary["turns"] == 1  # 2020 年那行被窗口滤掉
    assert summary["input_tokens"] == 5

    all_time = client.get("/v1/admin/usage/summary", headers=headers).json()
    assert all_time["turns"] == 2


def test_admin_users_rejects_unknown_window(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings=settings))
    admin = _register(client, "boss")
    headers = {"Authorization": f"Bearer {admin['token']}"}
    response = client.get(
        "/v1/admin/users", headers=headers, params={"window": "99h"}
    )
    assert response.status_code == 422
