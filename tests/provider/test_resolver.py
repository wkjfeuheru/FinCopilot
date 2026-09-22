import pytest

from finharness.config.crypto import SecretCipher
from finharness.config.settings import Settings
from finharness.config.store import ConfigStore
from finharness.provider.anthropic_compat import AnthropicCompatProvider
from finharness.provider.fake import FakeProvider
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.provider.resolver import NotConfigured, ProviderResolver

FAKE_KEY = "placeholder-value-a"


@pytest.fixture
def store(tmp_path):
    return ConfigStore(tmp_path / "config.db", cipher=SecretCipher(tmp_path / "secret.key"))


def build_resolver(store, settings):
    return ProviderResolver(store_factory=lambda: store, settings=settings)


def test_resolver_prefers_the_active_database_config(store, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-value")
    store.create(
        name="custom",
        kind="openai_compat",
        base_url="https://custom.example/v1",
        model="custom-model",
        env_key=None,
        api_key=FAKE_KEY,
        activate=True,
    )
    resolver = build_resolver(store, Settings())

    provider = resolver.current()

    assert isinstance(provider, OpenAICompatProvider)
    assert provider.base_url == "https://custom.example/v1"
    assert provider.model == "custom-model"
    assert provider.api_key == FAKE_KEY


def test_remote_resolver_rejects_legacy_active_non_preset_url(store):
    store.create(
        name="legacy-local",
        kind="openai_compat",
        base_url="https://169.254.169.254/v1",
        model="custom-model",
        env_key=None,
        api_key=FAKE_KEY,
        activate=True,
    )
    settings = Settings(server={"host": "0.0.0.0", "allow_remote": True})
    resolver = build_resolver(store, settings)

    with pytest.raises(
        NotConfigured,
        match="远程部署只允许使用运维预设的 Provider 地址",
    ):
        resolver.current()


def test_resolver_falls_back_to_settings_preset_and_environment(store, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-value")
    resolver = build_resolver(store, Settings())

    provider = resolver.current()

    assert isinstance(provider, OpenAICompatProvider)
    assert provider.base_url == "https://api.deepseek.com/v1"
    assert provider.api_key == "env-value"


def test_resolver_raises_not_configured_without_any_key(store, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    resolver = build_resolver(store, Settings())

    with pytest.raises(NotConfigured):
        resolver.current()


def test_resolver_builds_anthropic_kind_from_the_database(store):
    store.create(
        name="kimi",
        kind="anthropic_compat",
        base_url="https://api.moonshot.cn/anthropic/v1",
        model="kimi-k2",
        env_key=None,
        api_key=FAKE_KEY,
        activate=True,
    )
    resolver = build_resolver(store, Settings())

    provider = resolver.current()

    assert isinstance(provider, AnthropicCompatProvider)
    assert provider.api_version == "2023-06-01"


def test_resolver_supports_explicit_fake_kind(store):
    store.create(
        name="offline",
        kind="fake",
        base_url=None,
        model="ignored",
        env_key=None,
        api_key=None,
        activate=True,
    )
    resolver = build_resolver(store, Settings())

    assert isinstance(resolver.current(), FakeProvider)


def test_resolver_uses_the_stored_key_before_the_environment(store, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-value")
    record = store.create(
        name="custom",
        kind="openai_compat",
        base_url="https://custom.example/v1",
        model="m",
        env_key="DEEPSEEK_API_KEY",
        api_key=FAKE_KEY,
        activate=True,
    )
    resolver = build_resolver(store, Settings())

    provider = resolver.current()

    assert provider.api_key == FAKE_KEY
    assert record.has_key is True


def test_resolver_caches_until_invalidated(store, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-value")
    resolver = build_resolver(store, Settings())

    first = resolver.current()
    second = resolver.current()

    assert first is second


def test_resolver_status_reports_database_source(store):
    store.create(
        name="custom",
        kind="openai_compat",
        base_url="https://custom.example/v1",
        model="m1",
        env_key=None,
        api_key=FAKE_KEY,
        activate=True,
    )
    resolver = build_resolver(store, Settings())

    status = resolver.status()

    assert status.configured is True
    assert status.source == "database"
    assert status.name == "custom"
    assert status.model == "m1"
    assert status.has_key is True


def test_resolver_status_marks_unconfigured_when_no_key_resolves(store, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    resolver = build_resolver(store, Settings())

    status = resolver.status()

    assert status.configured is False
    assert status.source == "settings"
