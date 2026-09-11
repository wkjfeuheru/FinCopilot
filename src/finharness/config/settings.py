"""Settings loading and validation for local FinHarness runs."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class SettingsError(ValueError):
    """Raised when configuration cannot safely start the requested service."""


@dataclass(slots=True)
class ThinkingSettings:
    enabled: bool = True
    budget_tokens: int = 2048


@dataclass(slots=True)
class ModelSettings:
    provider: str = "deepseek"
    model_name: str = "deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 4096
    thinking: ThinkingSettings = field(default_factory=ThinkingSettings)


@dataclass(slots=True)
class CostSettings:
    input: float = 1.0
    output: float = 2.0


@dataclass(slots=True)
class ProviderSettings:
    kind: str = "openai_compat"
    base_url: str = "https://api.deepseek.com/v1"
    env_key: str = "DEEPSEEK_API_KEY"
    api_version: str | None = None
    first_byte_timeout_s: float = 30.0
    idle_timeout_s: float = 60.0
    cost_per_1m: CostSettings = field(default_factory=CostSettings)


@dataclass(slots=True)
class PermissionSettings:
    default_mode: str = "default"


@dataclass(slots=True)
class ToolSettings:
    timeout_default_s: int = 30
    timeout_overrides: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DataSettings:
    adapter_order: list[str] = field(default_factory=lambda: ["akshare", "tushare", "baostock"])
    tushare_token_env: str = "TUSHARE_TOKEN"
    throttle_seconds: float = 1.0
    cache_dir: Path = field(default_factory=lambda: Path.cwd() / "data_cache")
    cache_ttl_days: dict[str, int] = field(default_factory=lambda: {"quote": 1, "kline": 1, "indicators": 7, "financials": 365, "announcements": 7})


@dataclass(slots=True)
class ContextSettings:
    max_turns: int = 30
    compaction_ratio: float = 0.8
    trim_rows: int = 20
    max_result_tokens: int = 1000
    max_tool_schema_tokens: int = 60
    short_mem_cap: int = 200
    recall_max_tokens: int = 400
    ltm_inject_max_tokens: int = 600


@dataclass(slots=True)
class AuditSettings:
    log_path: Path = field(default_factory=lambda: Path.cwd() / "logs" / "audit.jsonl")


@dataclass(slots=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    session_ttl_s: int = 1800
    confirm_ttl_s: int = 120
    static_dir: Path = field(default_factory=lambda: Path.cwd() / "src" / "finharness" / "server" / "static")
    allow_remote: bool = False


@dataclass(slots=True)
class PathSettings:
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "output")
    memory_db: Path = field(default_factory=lambda: Path.cwd() / "data_cache" / "memory.db")
    skills_dir: Path = field(default_factory=lambda: Path.cwd() / "skills")


def _merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = defaults.copy()
    for key, value in overrides.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _validate_keys(payload: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise SettingsError(f"Unknown configuration field in {location}: {sorted(unknown)[0]}")


def _object(payload: Any, allowed: set[str], location: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SettingsError(f"{location} must be an object")
    _validate_keys(payload, allowed, location)
    return payload


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _environment_overrides(payload: dict[str, Any]) -> None:
    for name in os.environ:
        if name.startswith("FINH_") and "__" in name:
            raise SettingsError(f"环境变量 {name} 不支持双下划线")
    mappings: dict[str, tuple[str, str, type[Any] | str]] = {
        "FINH_MODEL_PROVIDER": ("model", "provider", str),
        "FINH_MODEL_MODEL_NAME": ("model", "model_name", str),
        "FINH_SERVER_PORT": ("server", "port", int),
        "FINH_DATA_ADAPTER_ORDER": ("data", "adapter_order", "json_list"),
        "FINH_TOOLS_TIMEOUT_OVERRIDES": ("tools", "timeout_overrides", "json_dict"),
    }
    for name, (section, key, converter) in mappings.items():
        if name not in os.environ:
            continue
        raw = os.environ[name]
        try:
            if converter is int:
                value: Any = int(raw)
            elif converter == "json_list":
                value = json.loads(raw)
                if not isinstance(value, list):
                    raise ValueError("expected JSON list")
            elif converter == "json_dict":
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("expected JSON object")
            else:
                value = raw
        except (ValueError, json.JSONDecodeError) as exc:
            raise SettingsError(f"Invalid value for environment variable {name}") from exc
        payload.setdefault(section, {})[key] = value


@dataclass(slots=True)
class Settings:
    model: ModelSettings = field(default_factory=ModelSettings)
    providers: dict[str, ProviderSettings] = field(default_factory=lambda: {
        "deepseek": ProviderSettings(),
        "kimi": ProviderSettings(base_url="https://api.moonshot.cn/v1", env_key="MOONSHOT_API_KEY"),
        "glm": ProviderSettings(base_url="https://open.bigmodel.cn/api/paas/v4", env_key="ZHIPU_API_KEY"),
    })
    permission: PermissionSettings = field(default_factory=PermissionSettings)
    tools: ToolSettings = field(default_factory=ToolSettings)
    data: DataSettings = field(default_factory=DataSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    audit: AuditSettings = field(default_factory=AuditSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    paths: PathSettings = field(default_factory=PathSettings)

    @classmethod
    def from_file(cls, path: str | Path | None = None, *, require_api_key: bool = False) -> "Settings":
        file_path = Path(path) if path is not None else None
        base = (file_path.parent if file_path is not None else Path.cwd()).resolve()
        payload: dict[str, Any] = {}
        if file_path is not None and file_path.exists():
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SettingsError(f"Invalid JSON in {file_path}") from exc
        if not isinstance(payload, dict):
            raise SettingsError("Settings root must be an object")
        allowed = {"model", "providers", "permission", "tools", "data", "context", "audit", "server", "paths"}
        _validate_keys(payload, allowed, "root")
        payload = _merge({}, payload)
        _environment_overrides(payload)

        defaults = {
            "model": asdict(ModelSettings()), "permission": asdict(PermissionSettings()),
            "tools": asdict(ToolSettings()), "data": asdict(DataSettings()),
            "context": asdict(ContextSettings()), "audit": asdict(AuditSettings()),
            "server": asdict(ServerSettings()), "paths": asdict(PathSettings()),
        }
        defaults["data"]["cache_dir"] = Path("data_cache")
        defaults["audit"]["log_path"] = Path("logs/audit.jsonl")
        defaults["server"]["static_dir"] = Path("src/finharness/server/static")
        defaults["paths"].update({
            "output_dir": Path("output"), "memory_db": Path("data_cache/memory.db"), "skills_dir": Path("skills"),
        })
        for section, default in defaults.items():
            if section in payload:
                _object(payload[section], set(default), section)
        if "model" in payload and "thinking" in payload["model"]:
            _object(payload["model"]["thinking"], set(defaults["model"]["thinking"]), "model.thinking")
        if "data" in payload and "cache_ttl_days" in payload["data"]:
            _object(
                payload["data"]["cache_ttl_days"],
                set(defaults["data"]["cache_ttl_days"]),
                "data.cache_ttl_days",
            )

        providers_payload = payload.get("providers", {})
        if not isinstance(providers_payload, dict):
            raise SettingsError("providers must be an object")
        provider_defaults = asdict(ProviderSettings())
        provider_builtin = cls().providers
        providers: dict[str, ProviderSettings] = {}
        for name, values in providers_payload.items():
            _object(values, set(provider_defaults), f"providers.{name}")
            if "cost_per_1m" in values:
                _object(values["cost_per_1m"], set(provider_defaults["cost_per_1m"]), f"providers.{name}.cost_per_1m")
            base_provider = asdict(provider_builtin.get(name, ProviderSettings()))
            merged_provider = _merge(base_provider, values)
            providers[name] = ProviderSettings(**{**merged_provider, "cost_per_1m": CostSettings(**merged_provider["cost_per_1m"])})
        for name, default_provider in provider_builtin.items():
            providers.setdefault(name, default_provider)

        merged = {section: _merge(default, payload.get(section, {})) for section, default in defaults.items()}
        settings = cls(
            model=ModelSettings(**{**merged["model"], "thinking": ThinkingSettings(**merged["model"]["thinking"])}),
            providers=providers,
            permission=PermissionSettings(**merged["permission"]), tools=ToolSettings(**merged["tools"]),
            data=DataSettings(**{**merged["data"], "cache_dir": _resolve_path(merged["data"]["cache_dir"], base)}),
            context=ContextSettings(**merged["context"]), audit=AuditSettings(log_path=_resolve_path(merged["audit"]["log_path"], base)),
            server=ServerSettings(**{**merged["server"], "static_dir": _resolve_path(merged["server"]["static_dir"], base)}),
            paths=PathSettings(**{key: _resolve_path(value, base) for key, value in merged["paths"].items()}),
        )
        settings.validate(require_api_key=require_api_key)
        return settings

    def validate(self, *, require_api_key: bool = False) -> None:
        if self.model.provider != "fake" and self.model.provider not in self.providers:
            raise SettingsError(f"Unknown provider: {self.model.provider}")
        if self.model.provider == "fake":
            provider = None
        else:
            provider = self.providers[self.model.provider]
        if provider is not None:
            if provider.kind not in {"openai_compat", "anthropic_compat"}:
                raise SettingsError(f"providers.{self.model.provider}.kind must be openai_compat or anthropic_compat")
            parsed = urlparse(provider.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise SettingsError(f"providers.{self.model.provider}.base_url must be an absolute HTTP(S) URL")
            if provider.first_byte_timeout_s <= 0 or provider.idle_timeout_s <= 0:
                raise SettingsError(f"providers.{self.model.provider} timeout values must be positive")
        if self.permission.default_mode not in {"default", "plan", "auto"}:
            raise SettingsError("permission.default_mode must be default, plan, or auto")
        if not 1 <= self.server.port <= 65535:
            raise SettingsError("server.port must be between 1 and 65535")
        if not 1 <= self.context.max_turns <= 100:
            raise SettingsError("context.max_turns must be between 1 and 100")
        if not self.data.adapter_order:
            raise SettingsError("data.adapter_order must not be empty")
        if not self.server.allow_remote and self.server.host not in {"127.0.0.1", "::1", "localhost"}:
            raise SettingsError("server.allow_remote is required for non-loopback host")
        if self.server.allow_remote and self.permission.default_mode != "default":
            raise SettingsError("permission.default_mode must be default when server.allow_remote is enabled")
        if require_api_key and self.model.provider != "fake":
            provider = self.providers[self.model.provider]
            if not os.getenv(provider.env_key):
                raise SettingsError(f"Missing API key for provider {self.model.provider}: set environment variable {provider.env_key}.")

    def validate_runtime(self) -> None:
        self.validate(require_api_key=True)
        parent = self.audit.log_path.parent
        if not parent.exists() or not parent.is_dir() or not os.access(parent, os.W_OK):
            raise SettingsError(f"Audit log parent directory is not writable: {parent}")
