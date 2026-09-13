import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from finharness.config.settings import Settings, SettingsError


def write_settings(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_settings_defaults_include_complete_contract(tmp_path):
    settings = Settings.from_file(write_settings(tmp_path, {}))

    assert settings.model.provider == "deepseek"
    assert settings.model.model_name == "deepseek-chat"
    assert settings.model.thinking.enabled is True
    assert settings.providers["deepseek"].kind == "openai_compat"
    assert settings.permission.default_mode == "default"
    assert settings.tools.timeout_default_s == 30
    assert settings.data.adapter_order == ("akshare", "tushare", "baostock")
    assert settings.context.max_turns == 30
    assert settings.server.confirm_ttl_s == 120
    assert settings.server.allow_remote is False
    assert set(settings.providers) >= {"deepseek", "kimi", "glm", "volcano", "qwen", "fake"}
    assert settings.providers["kimi"].kind == "anthropic_compat"
    assert settings.providers["kimi"].base_url == "https://api.moonshot.cn/anthropic/v1"
    assert settings.providers["glm"].kind == "anthropic_compat"
    assert settings.providers["glm"].base_url == "https://open.bigmodel.cn/api/anthropic/v1"
    assert settings.providers["glm"].env_key == "ZHIPU_API_KEY"
    assert settings.providers["volcano"].kind == "openai_compat"
    assert settings.providers["volcano"].base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert settings.providers["volcano"].env_key == "ARK_API_KEY"
    assert settings.providers["qwen"].kind == "openai_compat"
    assert settings.providers["qwen"].base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.providers["qwen"].env_key == "DASHSCOPE_API_KEY"
    assert settings.providers["fake"].kind == "fake"
    assert settings.providers["deepseek"].first_byte_timeout_s == 30.0
    assert settings.providers["deepseek"].idle_timeout_s == 60.0

def test_provider_validation_rejects_invalid_kind_url_and_timeout(tmp_path):
    cases = [
        ({"kind": "wat"}, "kind"),
        ({"base_url": "/relative"}, "base_url"),
        ({"first_byte_timeout_s": 0}, "timeout"),
        ({"idle_timeout_s": -1}, "timeout"),
    ]
    for override, message in cases:
        with pytest.raises(SettingsError, match=message):
            Settings.from_file(write_settings(tmp_path, {"providers": {"deepseek": override}}))

def test_only_selected_provider_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GLM_KEY", raising=False)
    path = write_settings(tmp_path, {"model": {"provider": "deepseek"}, "providers": {"glm": {"env_key": "GLM_KEY"}}})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ok")
    Settings.from_file(path, require_api_key=True)

def test_selected_glm_requires_its_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    path = write_settings(tmp_path, {"model": {"provider": "glm"}})
    with pytest.raises(SettingsError, match="ZHIPU_API_KEY"):
        Settings.from_file(path, require_api_key=True)

def test_invalid_unselected_provider_is_rejected_by_strict_schema(tmp_path):
    path = write_settings(tmp_path, {"providers": {"unused": {"kind": "invalid"}}})

    with pytest.raises(SettingsError, match="providers.unused.kind"):
        Settings.from_file(path)


def test_no_argument_loads_settings_json_from_working_directory(monkeypatch, tmp_path):
    write_settings(tmp_path, {"model": {"provider": "fake"}})
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_file()

    assert settings.model.provider == "fake"


def test_settings_loads_json_and_resolves_relative_paths(tmp_path):
    path = write_settings(
        tmp_path,
        {
            "model": {"provider": "deepseek", "temperature": 0.25},
            "data": {"adapter_order": ["akshare"], "cache_dir": "cache"},
            "audit": {"log_path": "logs/audit.jsonl"},
            "paths": {"output_dir": "artifacts", "memory_db": "state/memory.db", "skills_dir": "methodology"},
        },
    )

    settings = Settings.from_file(path)

    assert settings.model.temperature == 0.25
    assert settings.data.adapter_order == ("akshare",)
    assert settings.data.cache_dir == (tmp_path / "cache").resolve()
    assert settings.audit.log_path == (tmp_path / "logs" / "audit.jsonl").resolve()
    assert settings.paths.output_dir == (tmp_path / "artifacts").resolve()
    assert settings.paths.memory_db == (tmp_path / "state" / "memory.db").resolve()
    assert settings.paths.skills_dir == (tmp_path / "methodology").resolve()


def test_default_relative_paths_use_the_configuration_file_parent(tmp_path):
    settings = Settings.from_file(write_settings(tmp_path, {}))

    assert settings.data.cache_dir == (tmp_path / "data_cache").resolve()
    assert settings.audit.log_path == (tmp_path / "logs" / "audit.jsonl").resolve()
    assert settings.server.static_dir == (tmp_path / "src" / "finharness" / "server" / "static").resolve()
    assert settings.paths.memory_db == (tmp_path / "data_cache" / "memory.db").resolve()


def test_provider_overrides_deep_merge_preset_defaults(tmp_path):
    settings = Settings.from_file(write_settings(tmp_path, {"providers": {"deepseek": {"env_key": "ALT_KEY"}}}))

    assert settings.providers["deepseek"].env_key == "ALT_KEY"
    # Untouched preset fields survive the merge.
    assert settings.providers["deepseek"].kind == "openai_compat"
    assert settings.providers["deepseek"].base_url == "https://api.deepseek.com/v1"
    assert settings.providers["deepseek"].idle_timeout_s == 60.0


def test_settings_environment_overrides_scalar_and_json_fields(monkeypatch, tmp_path):
    path = write_settings(tmp_path, {"model": {"provider": "fake"}})
    monkeypatch.setenv("FINH_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("FINH_SERVER_PORT", "8123")
    monkeypatch.setenv("FINH_DATA_ADAPTER_ORDER", '["akshare"]')
    monkeypatch.setenv("FINH_TOOLS_TIMEOUT_OVERRIDES", '{"get_kline": 45}')

    settings = Settings.from_file(path)

    assert settings.model.provider == "deepseek"
    assert settings.server.port == 8123
    assert settings.data.adapter_order == ("akshare",)
    assert settings.tools.timeout_overrides == {"get_kline": 45}


def test_single_underscore_environment_overrides_nested_and_path_fields(monkeypatch, tmp_path):
    path = write_settings(tmp_path, {"model": {"provider": "fake"}})
    monkeypatch.setenv("FINH_MODEL_THINKING_ENABLED", "false")
    monkeypatch.setenv("FINH_SERVER_ALLOW_REMOTE", "true")
    monkeypatch.setenv("FINH_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("FINH_PATHS_OUTPUT_DIR", "generated")

    settings = Settings.from_file(path)

    assert settings.model.thinking.enabled is False
    assert settings.server.allow_remote is True
    assert settings.server.host == "0.0.0.0"
    assert settings.paths.output_dir == (tmp_path / "generated").resolve()


def test_json_environment_can_replace_provider_table(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "FINH_PROVIDERS",
        json.dumps(
            {
                "custom": {
                    "kind": "openai_compat",
                    "base_url": "https://models.example/v1",
                    "env_key": "CUSTOM_API_KEY",
                }
            }
        ),
    )
    path = write_settings(tmp_path, {"model": {"provider": "custom"}})

    settings = Settings.from_file(path)

    assert settings.providers["custom"].base_url == "https://models.example/v1"
    assert "deepseek" not in settings.providers


def test_double_underscore_environment_variables_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("FINH_MODEL__PROVIDER", "fake")

    with pytest.raises(SettingsError, match="双下划线"):
        Settings.from_file(write_settings(tmp_path, {}))


def test_unknown_prefixed_environment_variable_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("FINH_MODEL_TYPO", "deepseek")

    with pytest.raises(SettingsError, match="FINH_MODEL_TYPO"):
        Settings.from_file(write_settings(tmp_path, {}))


@pytest.mark.parametrize(
    "payload",
    [
        {"server": {"port": "8000"}},
        {"server": {"allow_remote": 0}},
        {"model": {"temperature": "0.1"}},
    ],
)
def test_json_values_are_strictly_typed(tmp_path, payload):
    with pytest.raises(SettingsError):
        Settings.from_file(write_settings(tmp_path, payload))


def test_settings_and_nested_models_are_frozen(tmp_path):
    settings = Settings.from_file(write_settings(tmp_path, {}))

    with pytest.raises(ValidationError, match="frozen"):
        settings.server.port = 9000

    with pytest.raises(AttributeError):
        settings.data.adapter_order.append("custom")

    with pytest.raises(TypeError):
        settings.tools.timeout_overrides["get_quote"] = 10

    with pytest.raises(TypeError):
        settings.providers["custom"] = settings.providers["deepseek"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"model": {"unknown": True}}, "unknown"),
        ({"permission": {"default_mode": "unsafe"}}, "default_mode"),
        ({"server": {"port": 70000}}, "port"),
        ({"context": {"max_turns": 101}}, "max_turns"),
        ({"data": {"adapter_order": []}}, "adapter_order"),
    ],
)
def test_invalid_configuration_fails_during_load(tmp_path, payload, message):
    with pytest.raises(SettingsError, match=message):
        Settings.from_file(write_settings(tmp_path, payload))


def test_unknown_cache_ttl_field_is_rejected(tmp_path):
    with pytest.raises(SettingsError, match="cache_ttl_days"):
        Settings.from_file(
            write_settings(tmp_path, {"data": {"cache_ttl_days": {"typo": 1}}})
        )


def test_missing_selected_provider_is_rejected(tmp_path):
    with pytest.raises(SettingsError, match="provider"):
        Settings.from_file(write_settings(tmp_path, {"model": {"provider": "missing"}}))


def test_remote_binding_requires_default_permission_mode(tmp_path):
    with pytest.raises(SettingsError, match="permission.default_mode"):
        Settings.from_file(
            write_settings(
                tmp_path,
                {
                    "server": {"host": "0.0.0.0", "allow_remote": True},
                    "permission": {"default_mode": "plan"},
                },
            )
        )


def test_local_binding_rejects_non_loopback_host(tmp_path):
    with pytest.raises(SettingsError, match="allow_remote"):
        Settings.from_file(write_settings(tmp_path, {"server": {"host": "0.0.0.0"}}))


def test_deepseek_requires_api_key_at_runtime(monkeypatch, tmp_path):
    path = write_settings(tmp_path, {})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(SettingsError, match="DEEPSEEK_API_KEY"):
        Settings.from_file(path, require_api_key=True)


def test_fake_provider_does_not_require_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = Settings.from_file(
        write_settings(tmp_path, {"model": {"provider": "fake"}}),
        require_api_key=True,
    )

    assert settings.model.provider == "fake"


def test_runtime_validation_checks_existing_audit_parent_without_creating_it(monkeypatch, tmp_path):
    audit_parent = tmp_path / "logs"
    audit_parent.mkdir()
    path = write_settings(tmp_path, {"audit": {"log_path": "logs/audit.jsonl"}})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    settings = Settings.from_file(path)
    settings.validate_runtime()

    assert not (tmp_path / "new-directory").exists()


def test_settings_loading_does_not_create_configured_directories(tmp_path):
    path = write_settings(tmp_path, {"data": {"cache_dir": "cache"}, "audit": {"log_path": "logs/audit.jsonl"}})

    Settings.from_file(path)

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "logs").exists()


# --- web search configuration (docs 03.4) -------------------------------------

def test_search_defaults_to_tavily_with_an_env_key(tmp_path):
    settings = Settings.from_file(write_settings(tmp_path, {"model": {"provider": "fake"}}))

    assert settings.search.kind == "tavily"
    assert settings.search.base_url == "https://api.tavily.com"
    # The key lives in the environment; no secret field exists on the model.
    assert settings.search.env_key == "TAVILY_API_KEY"
    assert not hasattr(settings.search, "api_key")


def test_search_section_is_configurable(tmp_path):
    path = write_settings(
        tmp_path,
        {
            "model": {"provider": "fake"},
            "search": {"env_key": "MY_SEARCH_KEY", "timeout_s": 12.5},
        },
    )

    settings = Settings.from_file(path)

    assert settings.search.env_key == "MY_SEARCH_KEY"
    assert settings.search.timeout_s == 12.5


def test_search_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("FINH_SEARCH_ENV_KEY", "FROM_ENV_KEY")
    monkeypatch.setenv("FINH_SEARCH_TIMEOUT_S", "7.5")
    path = write_settings(tmp_path, {"model": {"provider": "fake"}})

    settings = Settings.from_file(path)

    assert settings.search.env_key == "FROM_ENV_KEY"
    assert settings.search.timeout_s == 7.5


def test_unknown_search_backend_is_rejected(tmp_path):
    path = write_settings(
        tmp_path,
        {"model": {"provider": "fake"}, "search": {"kind": "not-a-search-engine"}},
    )

    with pytest.raises(SettingsError):
        Settings.from_file(path)


def test_web_has_its_own_cache_ttl(tmp_path):
    settings = Settings.from_file(write_settings(tmp_path, {"model": {"provider": "fake"}}))

    assert settings.data.cache_ttl_days["web"] == 1
