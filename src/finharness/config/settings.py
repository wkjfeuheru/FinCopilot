"""严格、只读且无副作用的 FinHarness 配置契约（门面）。

``Settings`` 本体与跨节校验、启动校验在这里；嵌套模型见 ``settings_models``，
加载与 ``FINH_`` 环境变量解析见 ``settings_loading``。三者的公开名都从本模块
重新导出，因此 ``from finharness.config.settings import Settings`` 等既有导入
不受拆分影响。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlparse

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from finharness.config.settings_loading import (
    _apply_environment_overrides,
    _deep_merge,
    _format_validation_error,
    _path_within,
    _read_json_object,
    _resolve_paths,
    _to_mutable,
)

# 这些模型定义在 ``settings_models``；此处重新导出，使既有
# ``from finharness.config.settings import <模型>`` 继续可用。
from finharness.config.settings_models import (
    AuditSettings,
    AuthSettings,
    ComputeSettings,
    ContextSettings,
    DataSettings,
    EmbeddingSettings,
    FrozenModel,
    FuyaoSettings,
    LoggingSettings,
    LtmSettings,
    MetricsSettings,
    ModelSettings,
    ObservabilitySettings,
    PathSettings,
    PermissionSettings,
    ProviderSettings,
    QuotaSettings,
    SearchSettings,
    ServerSettings,
    SettingsError,
    ThinkingSettings,
    ToolSettings,
    TraceStoreSettings,
    TracingSettings,
    VectorDbSettings,
    _default_providers,
)

__all__ = [
    "AuditSettings",
    "AuthSettings",
    "ComputeSettings",
    "ContextSettings",
    "DataSettings",
    "EmbeddingSettings",
    "FrozenModel",
    "FuyaoSettings",
    "LoggingSettings",
    "LtmSettings",
    "MetricsSettings",
    "ModelSettings",
    "ObservabilitySettings",
    "PathSettings",
    "PermissionSettings",
    "ProviderSettings",
    "QuotaSettings",
    "SearchSettings",
    "ServerSettings",
    "Settings",
    "SettingsError",
    "ThinkingSettings",
    "ToolSettings",
    "TraceStoreSettings",
    "TracingSettings",
    "VectorDbSettings",
]


def _module_available(name: str) -> bool:
    """可选依赖是否已安装；用于观测后端的启动期校验。"""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


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
    ltm: LtmSettings = Field(default_factory=LtmSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    quota: QuotaSettings = Field(default_factory=QuotaSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    fuyao: FuyaoSettings = Field(default_factory=FuyaoSettings)

    @field_validator("providers")
    @classmethod
    def freeze_providers(
        cls, value: Mapping[str, ProviderSettings]
    ) -> Mapping[str, ProviderSettings]:
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def validate_state_separation(self) -> Settings:
        """任何构造路径都必须满足状态与 agent 可达目录的分离。

        刻意做成模型级校验而不是只放在 ``validate()`` 里：后者只在
        ``from_file()`` 被调用时才跑，任何直接构造 ``Settings`` 的调用方
        （新入口、脚本、测试替身）都能静默绕过。安全约束一旦能被"忘记调用
        某个方法"绕过，就只是约定而非边界。
        """
        self._validate_state_is_not_agent_reachable()
        return self

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
    ) -> Settings:
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
        if self.server.allow_remote:
            for name, configured_provider in self.providers.items():
                if configured_provider.kind == "fake":
                    continue
                parsed = urlparse(configured_provider.base_url or "")
                if parsed.scheme != "https":
                    raise SettingsError(
                        f"Provider {name}：远程部署的 Provider 地址必须使用 HTTPS"
                    )
            if not self.auth.secure_cookie:
                raise SettingsError(
                    "server.allow_remote=true 时 auth.secure_cookie 必须为 true"
                )
            worker_url = self.compute.remote_worker_url
            parsed_worker = urlparse(worker_url or "")
            if parsed_worker.scheme not in {"http", "https"} or not parsed_worker.netloc:
                raise SettingsError(
                    "server.allow_remote=true 时 compute.remote_worker_url 必须是远程 worker 地址"
                )
            # /metrics 在启用时是公开端点（编排器要探测 health，但指标是运营数据）。
            # 远程暴露下必须带令牌，否则一个公网可读的 Prometheus 端点会泄露
            # 工具失败率、请求量与模型名等运营信息（隔离方案 G14）。
            if self.observability.metrics.enabled and not self.server.metrics_token:
                raise SettingsError(
                    "server.allow_remote=true 且启用 metrics 时必须设置 "
                    "server.metrics_token（/metrics 不得公网无鉴权可读）"
                )
        if require_api_key and provider.kind != "fake":
            env_key = provider.env_key
            if env_key is None or not os.getenv(env_key):
                raise SettingsError(
                    f"Provider {self.model.provider} 缺少 API Key；"
                    f"请设置环境变量 {env_key or '<未配置 env_key>'}"
                )

    def _validate_state_is_not_agent_reachable(self) -> None:
        """确保密钥与租户数据库不在 agent 可达的目录内。

        agent 可达目录是 ``paths.output_dir`` 与 ``data.cache_dir``：文件工具与
        产物下载端点都开放其中的子树（output 整体、cache 的 parquet/pdf），而
        cache 的文件名又是由请求形状决定的哈希。因此只要 ``secret.key``、
        ``users.db``、``memory.db`` 或 ``config.db`` 落在其中任何一个里，
        "读一个普通缓存文件"与"读到全部租户的供应商 key 与对话"之间就只隔一次
        包含性检查——这正是文档拒绝发布 ``run_python`` 时所依赖的那条论证，
        而它对任何进程内读写路径同样成立。

        这是启动即失败的硬约束，不做降级：把它做成警告，等于让多租户部署默认
        运行在一个已知可越权的布局上。
        """
        roots = {
            "paths.output_dir": self.paths.output_dir,
            "data.cache_dir": self.data.cache_dir,
        }
        state_files = {
            "paths.memory_db": self.paths.memory_db,
            "paths.auth_db": self.paths.auth_db,
            "paths.config_db": self.paths.config_db,
            "paths.secret_key": self.paths.secret_key,
            "paths.usage_db": self.paths.usage_db,
            "paths.compute_jobs_db": self.paths.compute_jobs_db,
            "paths.compute_packages_dir": self.paths.compute_packages_dir,
        }
        for state_name, state_path in state_files.items():
            for root_name, root in roots.items():
                if _path_within(state_path, root):
                    raise SettingsError(
                        f"{state_name} 位于 agent 可达目录 {root_name} 内："
                        f"{state_path}。密钥与租户数据库必须放在 paths.state_dir "
                        "下（默认 state/），否则缓存/产物读取路径可以触及它们。"
                    )

    def validate_runtime(self, *, check_audit: bool = True, require_api_key: bool = True) -> None:
        """校验启动所需外部条件，但不创建目录或访问网络。

        观测相关的检查刻意在这里做"快速失败"：``metrics``/``tracing`` 打开了
        开关却没有装上可选依赖或缺少凭据时，宁可启动即报错，也不要让运维方
        以为正在采集、实际上什么都没上报。
        """

        self.validate(require_api_key=require_api_key)
        self._validate_observability()
        # 旧布局（data_cache/）里的账号与历史若还在，而当前位置为空，必须当场
        # 失败：继续运行会得到一个看起来正常的空实例，"搬家中断"与"全新安装"
        # 在外部完全一样，运维方很难联想到原因（P0-3 的配套迁移）。
        from finharness.config.state_migration import check_for_unmigrated_state

        check_for_unmigrated_state(self)
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
