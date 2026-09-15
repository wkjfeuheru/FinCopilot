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


def _module_available(name: str) -> bool:
    """可选依赖是否已安装；用于观测后端的启动期校验。"""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
CacheKind = Literal["quote", "kline", "indicators", "financials", "announcements", "web", "reports", "macro", "industry"]
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
            # 研报元数据每日发布，但给定时间窗口的结果集在一天内是稳定的；
            # 全文则完全不变。
            "reports": 1,
            # 宏观序列按月/季度更新，因此缓存一周几乎不损失新鲜度，
            # 还能省去重复的上游调用。
            "macro": 7,
            # 行业指数历史与成分股随市场变化。
            "industry": 1,
            # 网页结果很快过期；缓存一天与新闻一致，
            # 并让同一会话内重复查询同一关键字无需再次请求。
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
    # 模型可用的输入窗口。当达到 compaction_ratio × context_window_tokens 时
    # 触发压缩。Provider 不会公布窗口大小，因此由配置指定；默认值适用于
    # deepseek-chat，其它模型应显式设置。
    context_window_tokens: PositiveInt = 64000
    trim_rows: PositiveInt = 20
    max_result_tokens: PositiveInt = 1000
    max_tool_schema_tokens: PositiveInt = 60
    # 循环防护：一模一样的 (tool, args) 调用重复达到此次数即被拒绝，若在提示后
    # 再次被拒绝则中止本次运行。取值保持较小，因为重复的相同调用从无信息量：
    # 其结果已在对话记录和缓存中。
    max_identical_tool_calls: Annotated[int, Field(ge=2, le=10)] = 3
    # 若计划的（修订号 + 步骤状态）连续这么多轮不变，即视为停滞；循环会注入一条
    # 建议修订的软提示。它从不阻断调用——只是让"运行没有推进"变得可见。
    plan_stall_turns: Annotated[int, Field(ge=1, le=20)] = 2
    # --- 记忆（docs 03.6.4）---
    # L2 事件环形缓冲区上限；超出时优先丢弃最旧的 *数据* 片段
    # （可从缓存恢复），而非结论。
    short_mem_cap: PositiveInt = 200
    recall_max_tokens: PositiveInt = 400
    ltm_inject_max_tokens: PositiveInt = 600
    # 预留给摘要片段的窗口占比；其余部分逐字保留近期轮次，
    # 因为它们描述的是当下正在发生的事。
    summary_budget_ratio: Annotated[float, Field(gt=0, le=0.5)] = 0.2
    # 无论预算如何，至少这么多近期轮次逐字保留。
    min_recent_rounds: Annotated[int, Field(ge=1, le=10)] = 2
    # 持久化对话的保留上限；先达到者触发清理。
    retention_conversations: PositiveInt = 200
    retention_days: PositiveInt = 180


class AuthSettings(FrozenModel):
    """注册登录（docs 03.13）：会话令牌 TTL 与 Cookie 属性。

    ``secure_cookie`` 默认 False 是因为默认部署绑定 127.0.0.1（HTTP）；
    对外经 HTTPS 暴露的部署应设为 true，使 Cookie 只经加密信道传输。
    """

    token_ttl_s: PositiveInt = 14 * 24 * 3600
    min_password_len: Annotated[int, Field(ge=8, le=128)] = 8
    secure_cookie: bool = False
    allow_register: bool = True


class AuditSettings(FrozenModel):
    log_path: Path = Path("logs/audit.jsonl")


class LoggingSettings(FrozenModel):
    """结构化日志（docs 03.14.1）。"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # 命名为 json_format 而非 json，避免遮蔽 pydantic BaseModel 的 ``json`` 方法。
    json_format: bool = True
    # None 表示只输出到 stdout；设置后额外追加一份 JSONL 文件。
    path: Path | None = None
    # 只有显式打开才记录完整的 Prompt/响应，且仍会先脱敏再截断。
    capture_payloads: bool = False
    max_payload_chars: PositiveInt = 4000


class MetricsSettings(FrozenModel):
    """Prometheus 指标（docs 03.14.2）；需要安装 ``observability`` extra。"""

    enabled: bool = False


class TracingSettings(FrozenModel):
    """LangSmith 追踪（docs 03.14.3）；需要安装 ``observability`` extra。"""

    enabled: bool = False
    backend: Literal["langsmith"] = "langsmith"
    project: str = "finharness"
    env_key: str = "LANGSMITH_API_KEY"
    # 追踪默认不附带 Prompt/响应；打开后仍会先脱敏。
    capture_payloads: bool = False


class ObservabilitySettings(FrozenModel):
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)


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
    # 用户与会话令牌存储（docs 03.13）；默认与其它库同住 data_cache/。
    auth_db: Path = Path("data_cache/users.db")
    # Skills 随包分发（docs 03.8）：以本文件为基准解析，
    # 这样无论工作目录如何都能找到目录清单。
    skills_dir: Path = Path(__file__).resolve().parent.parent / "skills"


class SearchSettings(FrozenModel):
    """外部网页搜索/抓取后端（docs 03.4）。

    配置方式与 Provider 类似：``kind`` 选择具体实现，``env_key`` 指定保存凭据的
    环境变量名。推荐来源是环境变量；``api_key`` 额外允许为 ``settings.json``
    未被跟踪（已被 git 忽略）的本地环境内联密钥，此时密钥以明文存储于磁盘。
    加载时缺少密钥并非错误——工具会在真正被调用时才报告 "search not configured"。
    """

    kind: Literal["tavily"] = "tavily"
    base_url: str = "https://api.tavily.com"
    env_key: str = "TAVILY_API_KEY"
    timeout_s: Annotated[float, Field(gt=0)] = 30.0
    api_key: str | None = None
    # 搜索 Provider 的出站代理。留空则使用系统代理：``httpx`` 只读取环境变量，
    # 而 A 股数据源（``requests``）还会遵循操作系统代理设置，因此在唯一出口
    # 是系统代理的机器上两者会不一致。此处显式设置的值会覆盖二者。
    proxy: str | None = None
    # 本地 PDF 回退：当托管提取器无法获取 ``.pdf`` 链接时，改从本机抓取。
    # 默认开启，因为对于 Provider 被阻断的文档这是唯一途径，且它仅限 PDF 链接
    # 并带有私有网段阻断；设为 false 可让所有抓取都走 Provider
    # （完全不发出本地出站请求）。
    local_pdf_fallback: bool = True


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
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
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
        """从 ``settings.json`` 与白名单 ``FINH_`` 环境变量加载并校验配置。

        参数:
            path: 配置文件路径；省略时使用当前工作目录下的 ``settings.json``。
            require_api_key: 为 True 时要求所选 Provider 具备可用的 API Key。
        返回:
            校验通过且不可变的 ``Settings`` 实例。
        异常:
            配置非法或校验失败时抛出 ``SettingsError``。
        """

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
        """校验跨字段约束：Provider 存在、服务器绑定与权限模式一致。

        ``require_api_key`` 为 True 时还要求所选 Provider 具备可用的 API Key。
        校验失败抛出 ``SettingsError``。
        """

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
        """校验启动所需外部条件，但不创建目录或访问网络。

        观测相关的检查刻意在这里做"快速失败"：``metrics``/``tracing`` 打开了
        开关却没有装上可选依赖或缺少凭据时，宁可启动即报错，也不要让运维方
        以为正在采集、实际上什么都没上报。
        """

        self.validate(require_api_key=require_api_key)
        self._validate_observability()
        if not check_audit:
            return
        parent = self.audit.log_path.parent
        if not parent.exists() or not parent.is_dir() or not os.access(parent, os.W_OK):
            raise SettingsError(
                f"审计日志父目录不可写：{parent}；请预先创建目录并授予写权限"
            )

    def _validate_observability(self) -> None:
        """校验观测后端的可用性：依赖、凭据与日志目录可写性。"""

        section = self.observability
        log_path = section.logging.path
        if log_path is not None:
            parent = log_path.parent
            if not parent.exists() or not parent.is_dir() or not os.access(parent, os.W_OK):
                raise SettingsError(
                    f"日志文件父目录不可写：{parent}；请预先创建目录或关闭 logging.path"
                )
        if section.metrics.enabled and not _module_available("prometheus_client"):
            raise SettingsError(
                "observability.metrics.enabled=true 但缺少 prometheus-client；"
                '请安装可选依赖：pip install -e ".[observability]"'
            )
        if not section.tracing.enabled:
            return
        if not _module_available("langsmith"):
            raise SettingsError(
                "observability.tracing.enabled=true 但缺少 langsmith；"
                '请安装可选依赖：pip install -e ".[observability]"'
            )
        env_key = section.tracing.env_key
        if not env_key or not os.getenv(env_key):
            raise SettingsError(
                f"observability.tracing.enabled=true 但环境变量 {env_key or '<未配置 env_key>'} 为空"
            )


def _read_json_object(file_path: Path) -> dict[str, Any]:
    """读取并解析 JSON object；文件不存在返回空 dict，非法 JSON 抛出 SettingsError。"""

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
    """递归合并两个字典，overrides 优先；不修改入参。"""

    result = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """按路径逐段创建或复用嵌套字典，并写入最终值。"""

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
    """将环境变量的原始字符串解析为注解声明的类型；复杂值要求合法 JSON。"""

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
    """把白名单中的 ``FINH_`` 环境变量解析后覆盖写入配置字典。"""

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
    """将配置中的相对路径按 ``base`` 解析为绝对路径。"""

    for path in (
        ("data", "cache_dir"),
        ("audit", "log_path"),
        ("observability", "logging", "path"),
        ("server", "static_dir"),
        ("paths", "output_dir"),
        ("paths", "memory_db"),
        ("paths", "auth_db"),
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
    """把 pydantic 校验错误整理为单行可读信息。"""

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
    "FINH_CONTEXT_PLAN_STALL_TURNS": (("context", "plan_stall_turns"), int),
    "FINH_CONTEXT_SUMMARY_BUDGET_RATIO": (("context", "summary_budget_ratio"), float),
    "FINH_CONTEXT_MIN_RECENT_ROUNDS": (("context", "min_recent_rounds"), int),
    "FINH_CONTEXT_RETENTION_CONVERSATIONS": (("context", "retention_conversations"), int),
    "FINH_CONTEXT_RETENTION_DAYS": (("context", "retention_days"), int),
    "FINH_CONTEXT_SHORT_MEM_CAP": (("context", "short_mem_cap"), int),
    "FINH_CONTEXT_RECALL_MAX_TOKENS": (("context", "recall_max_tokens"), int),
    "FINH_CONTEXT_LTM_INJECT_MAX_TOKENS": (("context", "ltm_inject_max_tokens"), int),
    "FINH_AUDIT_LOG_PATH": (("audit", "log_path"), Path),
    "FINH_OBSERVABILITY_LOGGING_LEVEL": (
        ("observability", "logging", "level"),
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
    ),
    "FINH_OBSERVABILITY_LOGGING_JSON_FORMAT": (("observability", "logging", "json_format"), bool),
    "FINH_OBSERVABILITY_LOGGING_PATH": (("observability", "logging", "path"), Path),
    "FINH_OBSERVABILITY_LOGGING_CAPTURE_PAYLOADS": (
        ("observability", "logging", "capture_payloads"),
        bool,
    ),
    "FINH_OBSERVABILITY_LOGGING_MAX_PAYLOAD_CHARS": (
        ("observability", "logging", "max_payload_chars"),
        int,
    ),
    "FINH_OBSERVABILITY_METRICS_ENABLED": (("observability", "metrics", "enabled"), bool),
    "FINH_OBSERVABILITY_TRACING_ENABLED": (("observability", "tracing", "enabled"), bool),
    "FINH_OBSERVABILITY_TRACING_BACKEND": (
        ("observability", "tracing", "backend"),
        Literal["langsmith"],
    ),
    "FINH_OBSERVABILITY_TRACING_PROJECT": (("observability", "tracing", "project"), str),
    "FINH_OBSERVABILITY_TRACING_ENV_KEY": (("observability", "tracing", "env_key"), str),
    "FINH_OBSERVABILITY_TRACING_CAPTURE_PAYLOADS": (
        ("observability", "tracing", "capture_payloads"),
        bool,
    ),
    "FINH_SERVER_HOST": (("server", "host"), str),
    "FINH_SERVER_PORT": (("server", "port"), int),
    "FINH_SERVER_SESSION_TTL_S": (("server", "session_ttl_s"), int),
    "FINH_SERVER_CONFIRM_TTL_S": (("server", "confirm_ttl_s"), int),
    "FINH_SERVER_STATIC_DIR": (("server", "static_dir"), Path),
    "FINH_SERVER_ALLOW_REMOTE": (("server", "allow_remote"), bool),
    "FINH_AUTH_TOKEN_TTL_S": (("auth", "token_ttl_s"), int),
    "FINH_AUTH_MIN_PASSWORD_LEN": (("auth", "min_password_len"), int),
    "FINH_AUTH_SECURE_COOKIE": (("auth", "secure_cookie"), bool),
    "FINH_AUTH_ALLOW_REGISTER": (("auth", "allow_register"), bool),
    "FINH_PATHS_OUTPUT_DIR": (("paths", "output_dir"), Path),
    "FINH_PATHS_MEMORY_DB": (("paths", "memory_db"), Path),
    "FINH_PATHS_AUTH_DB": (("paths", "auth_db"), Path),
    "FINH_PATHS_SKILLS_DIR": (("paths", "skills_dir"), Path),
    "FINH_SEARCH_KIND": (("search", "kind"), Literal["tavily"]),
    "FINH_SEARCH_BASE_URL": (("search", "base_url"), str),
    "FINH_SEARCH_ENV_KEY": (("search", "env_key"), str),
    "FINH_SEARCH_TIMEOUT_S": (("search", "timeout_s"), float),
    "FINH_SEARCH_API_KEY": (("search", "api_key"), str),
    "FINH_SEARCH_PROXY": (("search", "proxy"), str),
    "FINH_SEARCH_LOCAL_PDF_FALLBACK": (("search", "local_pdf_fallback"), bool),
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
