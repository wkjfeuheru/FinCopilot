"""server 层测试的共享辅助：注册并登录一个测试用户。

所有 /v1 端点都要求认证（docs 03.13），测试通过 ``Authorization:
Bearer`` 通道携带令牌。
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

DEFAULT_PASSWORD = "test-pass-123"


def register_and_login(
    client: TestClient, username: str | None = None, password: str = DEFAULT_PASSWORD
) -> tuple[str, dict[str, str]]:
    """注册（或登录）一个用户，返回 ``(token, headers)``。

    用户名不区分大小写地唯一；需要多用户的测试各自传不同的名字。
    """
    username = username or f"tester_{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/v1/auth/register", json={"username": username, "password": password}
    )
    if response.status_code == 409:
        response = client.post(
            "/v1/auth/login", json={"username": username, "password": password}
        )
    assert response.status_code == 200, response.text
    token = response.json()["token"]
    return token, {"Authorization": f"Bearer {token}"}


def authed_client(client: TestClient, username: str | None = None) -> TestClient:
    """在该客户端上注册一个用户并把 Bearer 头设为默认值。

    既有测试的每个请求无需逐个补头；多用户隔离测试用
    ``register_and_login`` 自行管理不同的令牌。注册出的用户身份
    挂在 ``client.finharness_user`` 上，供需要直接访问 store 的
    测试按用户作用域读写。
    """
    username = username or f"tester_{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/v1/auth/register", json={"username": username, "password": DEFAULT_PASSWORD}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    client.headers.update({"Authorization": f"Bearer {body['token']}"})
    client.finharness_user = body["user"]  # type: ignore[attr-defined]
    return client
