"""严格、只读且无副作用的 FinHarness 配置契约。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, get_args, get_origin
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class SettingsError(ValueError):
    """配置无法安全加载或启动时抛出。"""


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
CacheKind = Literal["quote", "kline", "indicators", "financials", "announcements", "web"]
ProviderKind = Literal["openai_compat", "anthropic_compat", "fake"]
PermissionMode = Literal["default", "plan", "auto"]


class FrozenModel(BaseModel):
    """所有嵌套配置共用的严格、冻结基类。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True)


class ThinkingSettings(FrozenModel):
    enabled: bool = True
    budget_tokens: PositiveInt = 2048


class ModelSettings(FrozenModel):
    provider: str = "deepseek"
    model_name: str = "deepseek-chat"
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.1
    max_tokens: PositiveInt = 4096
    thinking: ThinkingSettings = Field(default_factory=ThinkingSettings)


class ProviderSettings(FrozenModel):
    kind: ProviderKind = "openai_compat"
    base_url: str | None = None
    env_key: str | None = None
    api_version: str | None = None
    first_byte_timeout_s: Annotated[float, Field(gt=0)] = 30.0
    idle_timeout_s: Annotated[float, Field(gt=0)] = 60.0

    @model_validator(mode="after")
    def validate_endpoint_contract(self) -> "ProviderSettings":
        if self.kind == "fake":
            return self
        if not self.base_url:
            raise ValueError("base_url 是必填项")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url 必须是绝对 HTTP(S) URL")
        if not self.env_key or not self.env_key.strip():
            raise ValueError("env_key 是必填项")
        return self


class PermissionSettings(FrozenModel):
    default_mode: PermissionMode = "default"


class ToolSettings(FrozenModel):
    timeout_default_s: PositiveInt = 30
    timeout_overrides: Mapping[str, PositiveInt] = Field(default_factory=dict)
    resident: tuple[str, ...] = ()
    lazy: tuple[str, ...] = ()

    @field_validator("resident", "lazy", mode="before")
    @classmethod
    def freeze_tool_names(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("timeout_overrides")
    @classmethod
    def freeze_timeout_overrides(
        cls, value: Mapping[str, PositiveInt]
    ) -> Mapping[str, PositiveInt]:
        return MappingProxyType(dict(value))


class DataSettings(FrozenModel):
    adapter_order: Annotated[tuple[str, ...], Field(min_length=1)] = Field(
        default_factory=lambda: ("akshare", "tushare", "baostock")
    )
    tushare_token_env: str = "TUSHARE_TOKEN"
    throttle_seconds: NonNegativeFloat = 1.0
    cache_dir: Path = Path("data_cache")
    cache_ttl_days: Mapping[CacheKind, PositiveInt] = Field(
        default_factory=lambda: {
            "quote": 1,
            "kline": 1,
            "indicators": 7,
            "financials": 365,
            "announcements": 7,
            # Web results go stale fast; one day matches news and keeps a
            # within-session repeat of the same query free.
            "web": 1,
        }
    )
    @field_validator("adapter_order", mode="before")
    @classmethod
    def freeze_adapter_order(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("adapter_order")
    @classmethod
    def validate_adapter_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value):
            raise ValueError("adapter_order 不能包含空名称")
        return value

    @field_validator("cache_ttl_days")
    @classmethod
    def freeze_cache_ttls(
        cls, value: Mapping[CacheKind, PositiveInt]
    ) -> Mapping[CacheKind, PositiveInt]:
        return MappingProxyType(dict(value))


class ContextSettings(FrozenModel):
    max_turns: Annotated[int, Field(ge=1, le=100)] = 30
    compaction_ratio: Annotated[float, Field(gt=0, le=1)] = 0.8
    # The model's usable input window. Compaction fires at
    # compaction_ratio x context_window_tokens. Providers do not advertise a
    # window, so it is configuration; the default suits deepseek-chat and other
    # models should set it explicitly.
    context_window_tokens: PositiveInt = 64000
    trim_rows: PositiveInt = 20
    max_result_tokens: PositiveInt = 1000
    max_tool_schema_tokens: PositiveInt = 60
    # Loop guard: an identical (tool, args) call repeated this many times is
    # refused, and refused again after a nudge it aborts the run. Kept low
    # because a repeated identical call is never informative: its result is
    # already in the transcript and in the cache.
    max_identical_tool_calls: Annotated[int, Field(ge=2, le=10)] = 3
    # --- memory (docs 03.6.4) ---
    # L2 event ring cap; over it, the oldest *data* episodes are dropped first
    # (recoverable from the cache) rather than conclusions.
    short_mem_cap: PositiveInt = 200
    recall_max_tokens: PositiveInt = 400
    ltm_inject_max_tokens: PositiveInt = 600
    # Share of the window reserved for summary segments; the rest keeps recent
    # turns verbatim, because those describe what is happening right now.
    summary_budget_ratio: Annotated[float, Field(gt=0, le=0.5)] = 0.2
    # At least this many recent rounds stay verbatim regardless of the budget.
    min_recent_rounds: Annotated[int, Field(ge=1, le=10)] = 2
    # Retention bounds for persisted conversations; whichever comes first prunes.
    retention_conversations: PositiveInt = 200
    retention_days: PositiveInt = 180


class AuditSettings(FrozenModel):
    log_path: Path = Path("logs/audit.jsonl")


class ServerSettings(FrozenModel):
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    session_ttl_s: PositiveInt = 1800
    confirm_ttl_s: PositiveInt = 120
    static_dir: Path = Path("src/finharness/server/static")
    allow_remote: bool = False


class PathSettings(FrozenModel):
    output_dir: Path = Path("output")
    memory_db: Path = Path("data_cache/memory.db")
    # Skills ship with the package (docs 03.8): resolve from this file so the
    # catalogue is found regardless of the working directory.
    skills_dir: Path = Path(__file__).resolve().parent.parent / "skills"


class SearchSettings(FrozenModel):
    """External web search/fetch backend (docs 03.4).

    Configured like a provider: a ``kind`` picks the implementation and an
    ``env_key`` names the variable holding the credential. The recommended
    source is that environment variable; ``api_key`` additionally allows an
    inline key for local setups whose ``settings.json`` is untracked (it is
    git-ignored), in which case the secret sits in plaintext on disk. An absent
    key is not an error at load time — the tools report "search not configured"
    when actually called.
    """

    kind: Literal["tavily"] = "tavily"
    base_url: str = "https://api.tavily.com"
    env_key: str = "TAVILY_API_KEY"
    timeout_s: Annotated[float, Field(gt=0)] = 30.0
    api_key: str | None = None


def _default_providers() -> dict[str, ProviderSettings]:
    return {
        "deepseek": ProviderSettings(
            kind="openai_compat",
            base_url="https://api.deepseek.com/v1",
            env_key="DEEPSEEK_API_KEY",
        ),
        "kimi": ProviderSettings(
            kind="anthropic_compat",
            base_url="https://api.moonshot.cn/anthropic/v1",
            env_key="MOONSHOT_API_KEY",
        ),
        "glm": ProviderSettings(
            kind="anthropic_compat",
            base_url="https://open.bigmodel.cn/api/anthropic/v1",
            env_key="ZHIPU_API_KEY",
        ),
        "volcano": ProviderSettings(
            kind="openai_compat",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            env_key="ARK_API_KEY",
        ),
        "qwen": ProviderSettings(
            kind="openai_compat",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            env_key="DASHSCOPE_API_KEY",
        ),
        "fake": ProviderSettings(kind="fake"),
    }


class Settings(BaseSettings):
    """进程级配置；只从 ``settings.json`` 和显式 ``FINH_`` 映射加载。"""

    model_config = SettingsConfigDict(
        env_prefix="FINH_",
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )

    model: ModelSettings = Field(default_factory=ModelSettings)
    providers: Mapping[str, ProviderSettings] = Field(default_factory=_default_providers)
    permission: PermissionSettings = Field(default_factory=PermissionSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)

    @field_validator("providers")
    @classmethod
    def freeze_providers(
        cls, value: Mapping[str, ProviderSettings]
    ) -> Mapping[str, ProviderSettings]:
        return MappingProxyType(dict(value))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 环境变量由下方白名单解析，避免 pydantic-settings 的 ``__`` 隐式语义。
        return (init_settings,)

    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        *,
        require_api_key: bool = False,
    ) -> "Settings":
        file_path = Path(path) if path is not None else Path.cwd() / "settings.json"
        file_path = file_path.resolve()
        payload = _read_json_object(file_path)

        defaults = _to_mutable(cls())
        merged = _deep_merge(defaults, payload)
        _apply_environment_overrides(merged)
        _resolve_paths(merged, file_path.parent)

        try:
            settings = cls.model_validate(merged)
        except ValidationError as exc:
            raise SettingsError(_format_validation_error(exc)) from exc
        settings.validate(require_api_key=require_api_key)
        return settings

    def validate(self, *, require_api_key: bool = False) -> None:
        provider = self.providers.get(self.model.provider)
        if provider is None:
            raise SettingsError(f"未知 Provider：{self.model.provider}；请检查 model.provider")
        if not self.server.allow_remote and self.server.host not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise SettingsError("server.allow_remote=false 时 server.host 只能使用回环地址")
        if self.server.allow_remote and self.permission.default_mode != "default":
            raise SettingsError(
                "server.allow_remote=true 时 permission.default_mode 必须为 default"
            )
        if require_api_key and provider.kind != "fake":
            env_key = provider.env_key
            if env_key is None or not os.getenv(env_key):
                raise SettingsError(
                    f"Provider {self.model.provider} 缺少 API Key；"
                    f"请设置环境变量 {env_key or '<未配置 env_key>'}"
                )

    def validate_runtime(self, *, check_audit: bool = True, require_api_key: bool = True) -> None:
        """校验启动所需外部条件，但不创建目录或访问网络。"""

        self.validate(require_api_key=require_api_key)
        if not check_audit:
            return
        parent = self.audit.log_path.parent
        if not parent.exists() or not parent.is_dir() or not os.access(parent, os.W_OK):
            raise SettingsError(
                f"审计日志父目录不可写：{parent}；请预先创建目录并授予写权限"
            )


def _read_json_object(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"settings.json 不是合法 JSON：{file_path}") from exc
    except OSError as exc:
        raise SettingsError(f"无法读取配置文件：{file_path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise SettingsError("settings.json 根节点必须是 JSON object")
    return payload


def _to_mutable(value: Any) -> Any:
    """把冻结默认模型转换为可深合并的普通 Python 容器。"""

    if isinstance(value, BaseModel):
        return {
            name: _to_mutable(getattr(value, name))
            for name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {key: _to_mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_mutable(item) for item in value]
    return deepcopy(value)


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for segment in path[:-1]:
        child = target.get(segment)
        if not isinstance(child, dict):
            child = {}
            target[segment] = child
        target = child
    target[path[-1]] = value


def _is_raw_string(annotation: Any) -> bool:
    if annotation in {str, Path}:
        return True
    origin = get_origin(annotation)
    if origin is Literal:
        return all(isinstance(value, str) for value in get_args(annotation))
    return False


def _parse_environment_value(name: str, raw: str, annotation: Any) -> Any:
    try:
        if _is_raw_string(annotation):
            return raw
        if get_origin(annotation) in {type(None), Literal}:
            return raw
        return TypeAdapter(annotation, config=ConfigDict(strict=True)).validate_json(raw)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise SettingsError(
            f"环境变量 {name} 的值无效；复杂值必须使用合法 JSON"
        ) from exc


def _apply_environment_overrides(payload: dict[str, Any]) -> None:
    prefixed = {name.upper(): value for name, value in os.environ.items() if name.upper().startswith("FINH_")}
    for name in prefixed:
        if "__" in name:
            raise SettingsError(f"环境变量 {name} 不支持双下划线，请使用单下划线格式")
        if name not in _ENV_FIELDS:
            raise SettingsError(f"未知环境变量 {name}；请检查名称或移除它")

    for name, (path, annotation) in _ENV_FIELDS.items():
        if name not in prefixed:
            continue
        value = _parse_environment_value(name, prefixed[name], annotation)
        _set_nested(payload, path, value)


def _resolve_path_value(value: Any, base: Path) -> Any:
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_paths(payload: dict[str, Any], base: Path) -> None:
    for path in (
        ("data", "cache_dir"),
        ("audit", "log_path"),
        ("server", "static_dir"),
        ("paths", "output_dir"),
        ("paths", "memory_db"),
        ("paths", "skills_dir"),
    ):
        target = payload
        for segment in path[:-1]:
            child = target.get(segment)
            if not isinstance(child, dict):
                break
            target = child
        else:
            if path[-1] in target:
                target[path[-1]] = _resolve_path_value(target[path[-1]], base)


def _format_validation_error(exc: ValidationError) -> str:
    details: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        details.append(f"{location}: {error['msg']}")
    return "配置校验失败：" + "; ".join(details)


_ENV_FIELDS: dict[str, tuple[tuple[str, ...], Any]] = {
    "FINH_PROVIDERS": (("providers",), dict[str, ProviderSettings]),
    "FINH_MODEL_PROVIDER": (("model", "provider"), str),
    "FINH_MODEL_MODEL_NAME": (("model", "model_name"), str),
    "FINH_MODEL_TEMPERATURE": (("model", "temperature"), float),
    "FINH_MODEL_MAX_TOKENS": (("model", "max_tokens"), int),
    "FINH_MODEL_THINKING_ENABLED": (("model", "thinking", "enabled"), bool),
    "FINH_MODEL_THINKING_BUDGET_TOKENS": (("model", "thinking", "budget_tokens"), int),
    "FINH_PERMISSION_DEFAULT_MODE": (("permission", "default_mode"), PermissionMode),
    "FINH_TOOLS_TIMEOUT_DEFAULT_S": (("tools", "timeout_default_s"), int),
    "FINH_TOOLS_TIMEOUT_OVERRIDES": (("tools", "timeout_overrides"), dict[str, int]),
    "FINH_TOOLS_RESIDENT": (("tools", "resident"), list[str]),
    "FINH_TOOLS_LAZY": (("tools", "lazy"), list[str]),
    "FINH_DATA_ADAPTER_ORDER": (("data", "adapter_order"), list[str]),
    "FINH_DATA_TUSHARE_TOKEN_ENV": (("data", "tushare_token_env"), str),
    "FINH_DATA_THROTTLE_SECONDS": (("data", "throttle_seconds"), float),
    "FINH_DATA_CACHE_DIR": (("data", "cache_dir"), Path),
    "FINH_DATA_CACHE_TTL_DAYS": (("data", "cache_ttl_days"), dict[CacheKind, int]),
    "FINH_CONTEXT_MAX_TURNS": (("context", "max_turns"), int),
    "FINH_CONTEXT_COMPACTION_RATIO": (("context", "compaction_ratio"), float),
    "FINH_CONTEXT_CONTEXT_WINDOW_TOKENS": (("context", "context_window_tokens"), int),
    "FINH_CONTEXT_TRIM_ROWS": (("context", "trim_rows"), int),
    "FINH_CONTEXT_MAX_RESULT_TOKENS": (("context", "max_result_tokens"), int),
    "FINH_CONTEXT_MAX_TOOL_SCHEMA_TOKENS": (("context", "max_tool_schema_tokens"), int),
    "FINH_CONTEXT_MAX_IDENTICAL_TOOL_CALLS": (("context", "max_identical_tool_calls"), int),
    "FINH_CONTEXT_SUMMARY_BUDGET_RATIO": (("context", "summary_budget_ratio"), float),
    "FINH_CONTEXT_MIN_RECENT_ROUNDS": (("context", "min_recent_rounds"), int),
    "FINH_CONTEXT_RETENTION_CONVERSATIONS": (("context", "retention_conversations"), int),
    "FINH_CONTEXT_RETENTION_DAYS": (("context", "retention_days"), int),
    "FINH_CONTEXT_SHORT_MEM_CAP": (("context", "short_mem_cap"), int),
    "FINH_CONTEXT_RECALL_MAX_TOKENS": (("context", "recall_max_tokens"), int),
    "FINH_CONTEXT_LTM_INJECT_MAX_TOKENS": (("context", "ltm_inject_max_tokens"), int),
    "FINH_AUDIT_LOG_PATH": (("audit", "log_path"), Path),
    "FINH_SERVER_HOST": (("server", "host"), str),
    "FINH_SERVER_PORT": (("server", "port"), int),
    "FINH_SERVER_SESSION_TTL_S": (("server", "session_ttl_s"), int),
    "FINH_SERVER_CONFIRM_TTL_S": (("server", "confirm_ttl_s"), int),
    "FINH_SERVER_STATIC_DIR": (("server", "static_dir"), Path),
    "FINH_SERVER_ALLOW_REMOTE": (("server", "allow_remote"), bool),
    "FINH_PATHS_OUTPUT_DIR": (("paths", "output_dir"), Path),
    "FINH_PATHS_MEMORY_DB": (("paths", "memory_db"), Path),
    "FINH_PATHS_SKILLS_DIR": (("paths", "skills_dir"), Path),
    "FINH_SEARCH_KIND": (("search", "kind"), Literal["tavily"]),
    "FINH_SEARCH_BASE_URL": (("search", "base_url"), str),
    "FINH_SEARCH_ENV_KEY": (("search", "env_key"), str),
    "FINH_SEARCH_TIMEOUT_S": (("search", "timeout_s"), float),
    "FINH_SEARCH_API_KEY": (("search", "api_key"), str),
}


for _provider_name in ("DEEPSEEK", "KIMI", "GLM", "VOLCANO", "QWEN", "FAKE"):
    _provider_key = _provider_name.lower()
    _ENV_FIELDS.update(
        {
            f"FINH_PROVIDERS_{_provider_name}_KIND": (
                ("providers", _provider_key, "kind"),
                ProviderKind,
            ),
            f"FINH_PROVIDERS_{_provider_name}_BASE_URL": (
                ("providers", _provider_key, "base_url"),
                str,
            ),
            f"FINH_PROVIDERS_{_provider_name}_ENV_KEY": (
                ("providers", _provider_key, "env_key"),
                str,
            ),
            f"FINH_PROVIDERS_{_provider_name}_API_VERSION": (
                ("providers", _provider_key, "api_version"),
                str,
            ),
            f"FINH_PROVIDERS_{_provider_name}_FIRST_BYTE_TIMEOUT_S": (
                ("providers", _provider_key, "first_byte_timeout_s"),
                float,
            ),
            f"FINH_PROVIDERS_{_provider_name}_IDLE_TIMEOUT_S": (
                ("providers", _provider_key, "idle_timeout_s"),
                float,
            ),
        }
    )
