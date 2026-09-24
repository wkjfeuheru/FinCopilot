"""租户级数据生命周期：导出（可携带权）与抹除（遗忘权）。

隔离方案 Phase 2 的验收点：抹除必须覆盖**四处**——记忆库、向量库、产物/缓存
文件、审计 JSONL。只删数据库就会留下"看起来删干净了"的假象，因此这里的测试
逐处断言，而不是只看接口返回 200。
"""

from __future__ import annotations

import io
import json
import zipfile

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.server.api import create_app
from finharness.types import Msg


def _settings(tmp_path) -> Settings:
    return Settings(
        auth={"admin_bootstrap": True},
        audit={"log_path": tmp_path / "logs" / "audit.jsonl"},
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


def _admin_client(tmp_path) -> TestClient:
    """首个注册者即管理员（admin_bootstrap）。"""
    client = TestClient(create_app(settings=_settings(tmp_path)))
    response = client.post(
        "/v1/auth/register", json={"username": "boss", "password": "password-123"}
    )
    assert response.status_code == 200, response.text
    client.headers.update({"Authorization": f"Bearer {response.json()['token']}"})
    return client


def _seed_user_data(client: TestClient, user_id: str, tmp_path) -> None:
    """直接向三个存储写该租户的数据，覆盖抹除要碰的每一处。"""
    app = client.app
    app.state.memory_store.ensure_conversation("c_x", user_id=user_id)
    app.state.memory_store.append_messages("c_x", [Msg(role="user", content="hi")])
    app.state.memory_store.upsert_ltm_fact(
        user_id=user_id, key="risk", statement="偏好低风险", kind="preference"
    )
    # 产物与缓存文件（按租户命名空间）
    output = tmp_path / "output" / user_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text("# r", encoding="utf-8")
    cache = tmp_path / "cache" / "users" / user_id
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "payload.parquet").write_bytes(b"p")
    # 审计行（含该用户与其他用户各一行）
    audit = tmp_path / "logs" / "audit.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        json.dumps({"user_id": user_id, "action": "tool"}) + "\n"
        + json.dumps({"user_id": "someone_else", "action": "tool"}) + "\n",
        encoding="utf-8",
    )


def test_export_returns_a_zip_with_memory_and_files(tmp_path):
    client = _admin_client(tmp_path)
    user_id = client.get("/v1/auth/me").json()["user"]["id"]
    _seed_user_data(client, user_id, tmp_path)

    response = client.get("/v1/account/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        names = bundle.namelist()
        payload = json.loads(bundle.read("userdata.json"))
        assert any(name.startswith("output/") for name in names)
        assert any(name.startswith("cache/") for name in names)
    assert payload["user_id"] == user_id
    assert [f["key"] for f in payload["facts"]] == ["risk"]


def test_admin_purge_removes_all_four_surfaces(tmp_path):
    client = _admin_client(tmp_path)
    admin_id = client.get("/v1/auth/me").json()["user"]["id"]
    # 另建一个普通用户作为被抹除对象
    victim = client.post(
        "/v1/auth/register", json={"username": "victim", "password": "password-123"}
    ).json()["user"]["id"]
    _seed_user_data(client, victim, tmp_path)

    response = client.request("DELETE", f"/v1/admin/users/{victim}/data")

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["user_id"] == victim
    assert summary["deleted_rows"]["ltm_facts"] == 1
    assert summary["removed_files"] == 2
    assert summary["removed_audit_lines"] == 1
    # 1) 记忆库：该用户的对话与条目都不在了
    assert client.app.state.memory_store.list_conversations(user_id=victim) == []
    assert client.app.state.memory_store.list_ltm_facts(user_id=victim) == []
    # 2) 文件：产物与缓存目录都消失
    assert not (tmp_path / "output" / victim).exists()
    assert not (tmp_path / "cache" / "users" / victim).exists()
    # 3) 审计：仅该用户的行被移除，他人行保留
    audit_lines = (tmp_path / "logs" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
    assert json.loads(audit_lines[0])["user_id"] == "someone_else"
    # 管理员自身不受影响
    assert admin_id != victim


def test_admin_purge_refuses_a_missing_user(tmp_path):
    client = _admin_client(tmp_path)

    response = client.request("DELETE", "/v1/admin/users/does_not_exist/data")

    assert response.status_code == 404


def test_purge_requires_admin(tmp_path):
    client = _admin_client(tmp_path)
    # bootstrap 开着时 HTTP 注册一律是管理员，故普通用户直接落库（模拟关闭开关）。
    from finharness.auth.store import UserStore

    member = UserStore(tmp_path / "state" / "users.db").register("plain", "password-123")
    me = client.get("/v1/auth/me").json()["user"]["id"]
    # 清 cookie：注册响应 set-cookie 了管理员会话，而 resolve_user 是 cookie 优先。
    client.cookies.clear()

    response = client.request(
        "DELETE",
        f"/v1/admin/users/{me}/data",
        headers={"Authorization": f"Bearer {member.token}"},
    )

    assert response.status_code == 403
