"""注册登录：凭据校验、令牌通道与访问控制（docs 03.13）。"""

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app
from tests.server.conftest import register_and_login


def make_client(tmp_path, **overrides) -> TestClient:
    """认证测试专用客户端：用户库与记忆库都在临时目录内。"""
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "cache" / "memory.db",
            "auth_db": tmp_path / "cache" / "users.db",
        },
        **overrides,
    )
    return TestClient(create_app(FakeProvider(["ok"]), settings=settings))


# -- 注册 ---------------------------------------------------------------------

def test_register_returns_a_token_and_the_user_identity(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user"]["username"] == "alice"
    assert body["user"]["id"].startswith("u_")
    assert body["token"].startswith("t_")
    # 口令哈希绝不能出现在响应里。
    assert "password" not in response.text
    assert "pbkdf2" not in response.text


def test_register_sets_an_httponly_session_cookie(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/v1/auth/register", json={"username": "bob", "password": "secret-pass-1"}
    )

    cookie = response.headers.get("set-cookie", "")
    assert "finharness_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie.replace("samesite", "SameSite")


def test_duplicate_username_is_rejected_case_insensitively(tmp_path):
    client = make_client(tmp_path)
    client.post("/v1/auth/register", json={"username": "Carol", "password": "secret-pass-1"})

    response = client.post(
        "/v1/auth/register", json={"username": "carol", "password": "secret-pass-1"}
    )

    assert response.status_code == 409


def test_short_password_is_rejected(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/v1/auth/register", json={"username": "dave", "password": "short"}
    )

    assert response.status_code == 422
    assert "口令" in response.json()["detail"]


def test_too_short_username_is_rejected(tmp_path):
    client = make_client(tmp_path)

    response = client.post("/v1/auth/register", json={"username": "x", "password": "secret-pass-1"})

    assert response.status_code == 422


def test_registration_can_be_disabled(tmp_path):
    from finharness.config.settings import AuthSettings

    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "cache" / "memory.db",
            "auth_db": tmp_path / "cache" / "users.db",
        },
        auth=AuthSettings(allow_register=False),
    )
    client = TestClient(create_app(FakeProvider(["ok"]), settings=settings))

    response = client.post(
        "/v1/auth/register", json={"username": "eve", "password": "secret-pass-1"}
    )

    assert response.status_code == 403


# -- 登录 ---------------------------------------------------------------------

def test_login_with_correct_credentials_issues_a_new_token(tmp_path):
    client = make_client(tmp_path)
    first = client.post(
        "/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"}
    ).json()

    response = client.post(
        "/v1/auth/login", json={"username": "alice", "password": "secret-pass-1"}
    )

    assert response.status_code == 200
    assert response.json()["token"] != first["token"], "每次登录都签发新令牌"
    assert response.json()["user"]["id"] == first["user"]["id"]


def test_login_with_a_wrong_password_is_a_401(tmp_path):
    client = make_client(tmp_path)
    client.post("/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"})

    response = client.post(
        "/v1/auth/login", json={"username": "alice", "password": "wrong-pass-9"}
    )

    assert response.status_code == 401


def test_login_for_an_unknown_user_is_also_a_401(tmp_path):
    """不得让未知用户与错误口令可区分，否则可枚举用户名。"""
    client = make_client(tmp_path)

    response = client.post(
        "/v1/auth/login", json={"username": "nobody", "password": "secret-pass-1"}
    )

    assert response.status_code == 401


# -- me / logout --------------------------------------------------------------

def test_me_reports_the_user_behind_a_bearer_token(tmp_path):
    client = make_client(tmp_path)
    token, _ = register_and_login(client, "alice")

    response = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["user"]["username"] == "alice"


def test_me_works_via_the_session_cookie(tmp_path):
    """同源浏览器不发 Authorization 头；Cookie 通道必须同样可用。"""
    client = make_client(tmp_path)
    client.post("/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"})

    # TestClient 会保留注册时下发的 Cookie。
    response = client.get("/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["user"]["username"] == "alice"


def test_me_without_credentials_is_a_401(tmp_path):
    client = make_client(tmp_path)

    assert client.get("/v1/auth/me").status_code == 401


def test_logout_revokes_the_token(tmp_path):
    client = make_client(tmp_path)
    token, headers = register_and_login(client, "alice")
    assert client.get("/v1/auth/me", headers=headers).status_code == 200

    client.post("/v1/auth/logout", headers=headers)

    assert client.get("/v1/auth/me", headers=headers).status_code == 401


def test_an_unknown_token_is_a_401(tmp_path):
    client = make_client(tmp_path)

    response = client.get(
        "/v1/auth/me", headers={"Authorization": "Bearer t_deadbeef"}
    )

    assert response.status_code == 401


# -- 访问控制 -----------------------------------------------------------------

def test_protected_endpoints_require_authentication(tmp_path):
    client = make_client(tmp_path)

    for method, path in [
        ("get", "/v1/conversations"),
        ("get", "/v1/memory"),
        ("get", "/v1/tools"),
        ("get", "/v1/cache/stats"),
        ("get", "/v1/config"),
        ("post", "/v1/chat/stream"),
        ("post", "/v1/chat/respond"),
        ("post", "/v1/report"),
    ]:
        if method == "get":
            response = client.get(path)
        else:
            response = client.post(path, json={})
        assert response.status_code == 401, f"{method.upper()} {path} must require auth"


def test_health_stays_open(tmp_path):
    """存活探针不要求认证，否则编排器无法探测容器。"""
    client = make_client(tmp_path)

    assert client.get("/v1/health").status_code == 200


def test_strong_hashes_are_stored_not_plaintext(tmp_path):
    """库文件泄露时口令不能直接可读。"""
    client = make_client(tmp_path)
    register_and_login(client, "alice", password="secret-pass-1")

    store = client.app.state.user_store
    with store._connect() as connection:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE username = 'alice'"
        ).fetchone()
    assert row["password_hash"].startswith("pbkdf2_sha256$")
    assert "secret-pass-1" not in row["password_hash"]


def test_session_tokens_are_stored_hashed(tmp_path):
    """令牌本体只下发给客户端；库中仅存 SHA-256。"""
    client = make_client(tmp_path)
    token, _ = register_and_login(client, "alice")

    store = client.app.state.user_store
    with store._connect() as connection:
        hashes = [
            row["token_hash"]
            for row in connection.execute("SELECT token_hash FROM auth_sessions")
        ]
    assert token not in hashes
    assert all(len(value) == 64 for value in hashes)


def test_an_expired_token_is_rejected(tmp_path):
    client = make_client(tmp_path)
    token, headers = register_and_login(client, "alice")

    # 把到期时间拨到过去。
    store = client.app.state.user_store
    with store._connect() as connection:
        connection.execute("UPDATE auth_sessions SET expires_at = '2000-01-01T00:00:00+00:00'")

    assert client.get("/v1/auth/me", headers=headers).status_code == 401
