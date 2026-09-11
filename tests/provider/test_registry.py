import json

import pytest

from finharness.config.settings import SettingsError
from finharness.provider.registry import build_provider
import httpx


@pytest.mark.parametrize(
    ("preset", "base_url", "env_key", "model_name"),
    [
        ("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
        ("kimi", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "kimi-k2"),
        ("glm", "https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "glm-4.5"),
    ],
)
def test_build_provider_assembles_openai_compatible_presets(
    monkeypatch, tmp_path, preset, base_url, env_key, model_name
):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"model": {"provider": preset, "model_name": model_name}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(env_key, "offline-test-key")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    )
    provider = build_provider(path, client=client)
    try:
        assert provider.base_url == base_url
        assert provider.api_key == "offline-test-key"
        assert provider.model == model_name
    finally:
        import asyncio

        asyncio.run(client.aclose())


def test_build_provider_rejects_missing_deepseek_key(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(SettingsError, match="DEEPSEEK_API_KEY"):
        build_provider(path)

def test_build_provider_assembles_glm(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"model": {"provider": "glm"}}), encoding="utf-8")
    monkeypatch.setenv("ZHIPU_API_KEY", "glm-key")
    provider = build_provider(path, client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200))))
    assert provider.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert provider.api_key == "glm-key"
    assert provider.temperature == 0.1
    assert provider.max_tokens == 4096
    assert provider.first_byte_timeout_s == 30.0
    assert provider.idle_timeout_s == 60.0

def test_build_provider_assembles_anthropic(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"model": {"provider": "anthropic", "temperature": 0.3, "max_tokens": 123}, "providers": {"anthropic": {
        "kind": "anthropic_compat", "base_url": "https://api.anthropic.com", "env_key": "ANTHROPIC_API_KEY",
        "first_byte_timeout_s": 11.0, "idle_timeout_s": 22.0
    }}}), encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    provider = build_provider(path, client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200))))
    assert provider.api_version == "2023-06-01"
    assert provider.model == "deepseek-chat"
    assert provider.temperature == 0.3
    assert provider.max_tokens == 123
    assert provider.first_byte_timeout_s == 11.0
    assert provider.idle_timeout_s == 22.0

def test_build_provider_rejects_missing_glm_key(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"model": {"provider": "glm"}}), encoding="utf-8")
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    with pytest.raises(SettingsError, match="ZHIPU_API_KEY"):
        build_provider(path)
