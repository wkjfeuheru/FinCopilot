import json

import httpx
import pytest
from fastapi.testclient import TestClient

from finharness.config.crypto import SecretCipher
from finharness.config.settings import Settings
from finharness.config.store import ConfigStore
from finharness.server.api import create_app
from tests.server.conftest import authed_client

FAKE_KEY = "placeholder-value-a"


def _settings(tmp_path) -> Settings:
    """把用户库与记忆库都限定在临时目录，避免污染开发者的真实 store。"""
    return Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
    )


def _remote_settings(tmp_path) -> Settings:
    """构造远程部署设置，Provider 预设保持默认 HTTPS 地址。"""
    return Settings(
        data={"cache_dir": tmp_path / "cache"},
        server={"host": "0.0.0.0", "allow_remote": True},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
    )


@pytest.fixture
def store(tmp_path):
    return ConfigStore(tmp_path / "config.db", cipher=SecretCipher(tmp_path / "secret.key"))


@pytest.fixture
def client(store, tmp_path):
    app = create_app(settings=_settings(tmp_path), config_store=store)
    return authed_client(TestClient(app))


@pytest.fixture
def remote_client(store, tmp_path):
    app = create_app(settings=_remote_settings(tmp_path), config_store=store)
    return authed_client(TestClient(app))


def create_payload(**overrides):
    payload = {
        "name": "deepseek",
        "kind": "openai_compat",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_key": None,
        "api_key": FAKE_KEY,
        "activate": True,
    }
    payload.update(overrides)
    return payload


def test_get_config_reports_unconfigured_empty_state(client):
    response = client.get("/v1/config")

    assert response.status_code == 200
    body = response.json()
    assert body == {"configured": False, "active_id": None, "configs": []}


def test_create_config_never_echoes_the_secret(client):
    response = client.post("/v1/config", json=create_payload())

    assert response.status_code == 200
    body = response.json()["config"]
    assert body["name"] == "deepseek"
    assert body["has_key"] is True
    assert body["is_active"] is True
    assert FAKE_KEY not in response.text


def test_get_config_lists_redacted_records(client):
    client.post("/v1/config", json=create_payload())

    response = client.get("/v1/config")

    body = response.json()
    assert body["configured"] is True
    assert len(body["configs"]) == 1
    assert body["configs"][0]["has_key"] is True
    assert FAKE_KEY not in response.text
    assert body["active_id"] == body["configs"][0]["id"]


def test_create_config_validates_required_fields(client):
    response = client.post("/v1/config", json=create_payload(model="", base_url="not-a-url"))

    assert response.status_code == 422
    fields = {error["field"] for error in response.json()["detail"]["errors"]}
    assert fields == {"model", "base_url"}


def test_remote_create_rejects_non_preset_provider_url(remote_client):
    response = remote_client.post(
        "/v1/config",
        json=create_payload(base_url="https://169.254.169.254/v1"),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"] == [
        {
            "field": "base_url",
            "message": "远程部署只允许使用运维预设的 Provider 地址",
        }
    ]


def test_create_config_rejects_missing_key(client):
    response = client.post("/v1/config", json=create_payload(api_key=None, env_key=None))

    assert response.status_code == 422
    fields = {error["field"] for error in response.json()["detail"]["errors"]}
    assert fields == {"api_key"}


def test_create_config_rejects_duplicate_names(client):
    client.post("/v1/config", json=create_payload())

    response = client.post("/v1/config", json=create_payload(model="other"))

    assert response.status_code == 409


def test_activate_switches_configs_and_marks_single_active(client):
    first = client.post("/v1/config", json=create_payload(name="a")).json()["config"]
    second = client.post("/v1/config", json=create_payload(name="b", activate=False)).json()["config"]

    response = client.post(f"/v1/config/{second['id']}/activate")

    assert response.status_code == 200
    assert response.json()["config"]["is_active"] is True
    listing = client.get("/v1/config").json()
    assert listing["active_id"] == second["id"]
    active_flags = {c["id"]: c["is_active"] for c in listing["configs"]}
    assert active_flags[first["id"]] is False


def test_remote_activate_rejects_legacy_non_preset_provider_url(remote_client, store):
    legacy = store.create(
        name="legacy-local",
        kind="openai_compat",
        base_url="https://169.254.169.254/v1",
        model="deepseek-chat",
        env_key=None,
        api_key=FAKE_KEY,
        activate=False,
        user_id=remote_client.finharness_user["id"],
    )

    response = remote_client.post(f"/v1/config/{legacy.id}/activate")

    assert response.status_code == 422
    assert response.json()["detail"]["errors"] == [
        {
            "field": "base_url",
            "message": "远程部署只允许使用运维预设的 Provider 地址",
        }
    ]
    assert remote_client.get("/v1/config").json()["active_id"] is None


def test_update_without_key_keeps_the_stored_secret(client, store):
    created = client.post("/v1/config", json=create_payload()).json()["config"]

    response = client.put(
        f"/v1/config/{created['id']}",
        json=create_payload(model="deepseek-reasoner", api_key=None),
    )

    assert response.status_code == 200
    assert response.json()["config"]["model"] == "deepseek-reasoner"
    assert response.json()["config"]["has_key"] is True
    # 密钥按用户存储：直接查库要带上测试用户的归属。
    assert store.resolve_key(created["id"], user_id=client.finharness_user["id"]) == FAKE_KEY


def test_remote_update_cannot_switch_to_non_preset_provider_url(remote_client):
    created = remote_client.post("/v1/config", json=create_payload()).json()["config"]

    response = remote_client.put(
        f"/v1/config/{created['id']}",
        json=create_payload(base_url="https://169.254.169.254/v1", api_key=None),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"] == [
        {
            "field": "base_url",
            "message": "远程部署只允许使用运维预设的 Provider 地址",
        }
    ]


def test_delete_refuses_active_config_while_others_remain(client):
    first = client.post("/v1/config", json=create_payload(name="a")).json()["config"]
    client.post("/v1/config", json=create_payload(name="b", activate=False))

    response = client.delete(f"/v1/config/{first['id']}")

    assert response.status_code == 409


def test_delete_allows_non_active_and_last_config(client):
    first = client.post("/v1/config", json=create_payload(name="a")).json()["config"]
    second = client.post("/v1/config", json=create_payload(name="b", activate=False)).json()["config"]

    assert client.delete(f"/v1/config/{second['id']}").status_code == 200
    assert client.delete(f"/v1/config/{first['id']}").status_code == 200
    assert client.get("/v1/config").json()["configured"] is False


def test_missing_config_ids_return_404(client):
    assert client.post("/v1/config/999/activate").status_code == 404
    assert client.delete("/v1/config/999").status_code == 404
    assert client.put("/v1/config/999", json=create_payload()).status_code == 404


def test_presets_endpoint_lists_documented_and_custom_kinds(client):
    response = client.get("/v1/config/presets")

    assert response.status_code == 200
    names = {preset["name"] for preset in response.json()["presets"]}
    assert {"deepseek", "kimi", "glm"} <= names
    # fake 不作为预设提供（没有可填的内容）：即使默认 providers 表里存在，
    # 预设接口也不返回它——离线走 model.provider 显式选择或协议类型下拉。
    assert "fake" not in names
    deepseek = next(p for p in response.json()["presets"] if p["name"] == "deepseek")
    assert FAKE_KEY not in json.dumps(deepseek)


def _app_with_probe_transport(handler, tmp_path) -> object:
    """创建一个应用，其探测使用由 MockTransport 支撑的客户端。"""

    def client_factory(first_byte: float, idle: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return create_app(
        settings=_settings(tmp_path),
        config_store=ConfigStore(tmp_path / "probe.db", cipher=SecretCipher(tmp_path / "probe.key")),
        probe_client_factory=client_factory,
    )


def _probe_client(handler, tmp_path) -> TestClient:
    """带上认证的探测客户端。"""
    return authed_client(TestClient(_app_with_probe_transport(handler, tmp_path)))


def test_probe_reports_success_against_a_scripted_endpoint(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        line = json.dumps({"choices": [{"delta": {"content": "pong"}}]})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {line}\n\ndata: [DONE]\n\n".encode(),
        )

    probe_client = _probe_client(handler, tmp_path)

    response = probe_client.post(
        "/v1/config/probe",
        json={"kind": "openai_compat", "base_url": "https://example.test/v1", "model": "m", "api_key": FAKE_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["model"] == "m"


def test_probe_reports_auth_failure_as_clean_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    probe_client = _probe_client(handler, tmp_path)

    response = probe_client.post(
        "/v1/config/probe",
        json={"kind": "openai_compat", "base_url": "https://example.test/v1", "model": "m", "api_key": FAKE_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"]


def test_probe_requires_a_key(client):
    response = client.post(
        "/v1/config/probe",
        json={"kind": "openai_compat", "base_url": "https://example.test/v1", "model": "m"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "API Key" in body["error"]


def test_remote_probe_rejects_non_preset_provider_url_before_network(tmp_path):
    factory_calls: list[tuple[float, float]] = []
    transport_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(request)
        raise AssertionError("拒绝非法 Provider 地址前不应发起网络请求")

    def client_factory(first_byte: float, idle: float) -> httpx.AsyncClient:
        factory_calls.append((first_byte, idle))
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    app = create_app(
        settings=_remote_settings(tmp_path),
        config_store=ConfigStore(
            tmp_path / "probe.db", cipher=SecretCipher(tmp_path / "probe.key")
        ),
        probe_client_factory=client_factory,
    )
    probe_client = authed_client(TestClient(app))

    response = probe_client.post(
        "/v1/config/probe",
        json={
            "kind": "openai_compat",
            "base_url": "https://169.254.169.254/v1",
            "model": "m",
            "api_key": FAKE_KEY,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"] == [
        {
            "field": "base_url",
            "message": "远程部署只允许使用运维预设的 Provider 地址",
        }
    ]
    assert factory_calls == []
    assert transport_calls == []


def test_probe_accepts_fake_kind_without_network(client):
    response = client.post(
        "/v1/config/probe",
        json={"kind": "fake", "model": "ignored"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_chat_without_configuration_returns_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = create_app(
        settings=_settings(tmp_path),
        config_store=ConfigStore(tmp_path / "c.db", cipher=SecretCipher(tmp_path / "k.key")),
    )
    client = authed_client(TestClient(app, raise_server_exceptions=False))

    response = client.post("/v1/chat/stream", json={"message": "你好"})

    assert response.status_code == 400
    assert "配置" in response.json()["detail"]
