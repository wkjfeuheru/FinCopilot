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
NonNegativeInt = Annotated[int, Field(ge=0)]
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
    # 单条工具结果的 token 预算覆盖，键为工具名（docs 03.3.3）。与 timeout_overrides
    # 对称：不同工具的合理结果体量相差极大（一次报价 vs 一份研报全文），因此需要一个
    # 按工具调优的旋钮，而不是让所有工具共用一个数字。
    result_token_overrides: Mapping[str, PositiveInt] = Field(default_factory=dict)
    # 注册层级的运维覆盖。工具自己声明 ``@tool(tier=...)``，这两个列表只用于在不改代码的
    # 前提下调整口径（如临时把某个懒加载工具提为常驻）；留空表示完全听声明的。
    resident: tuple[str, ...] = ()
    lazy: tuple[str, ...] = ()
    # 已注入 schema 的上限。``tools`` 数组属于 provider 的缓存前缀，而按需激活会让它
    # 增长；设上限使"每轮携带多少 schema"成为可推理的常量，而非会话长度的函数。
    # 0 表示不限（沿用 docs 03.3.8 接受一次缓存失效的口径）。
    max_active: int = 0

    @field_validator("resident", "lazy", mode="before")
    @classmethod
    def freeze_tool_names(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("timeout_overrides", "result_token_overrides")
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
    # 压缩时每条工具结果进入摘要转录稿的 token 上限。此处过去是 compaction.py 里的
    # 字符硬编码 ``raw[:800]``：在一个按 token 计量的体系里，它带来的是一个更紧、
    # 且与语言无关的第二个上限——一条刚在 1000 token 预算下幸存的结果，会被悄悄砍到
    # 约 470 token。用 token 设置替代表达同一意图，并让它可被运维调整。
    compaction_result_tokens: PositiveInt = 500
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
    # 摘要层注入上限。该键此前名为 ltm_inject_max_tokens，但唯一消费者一直是
    # 摘要层（按对话隔离）；ltm 前缀如今让给真正的跨对话长期记忆（LtmSettings）。
    summary_inject_max_tokens: PositiveInt = 600
    # 预留给摘要片段的窗口占比；其余部分逐字保留近期轮次，
    # 因为它们描述的是当下正在发生的事。
    summary_budget_ratio: Annotated[float, Field(gt=0, le=0.5)] = 0.2
    # 无论预算如何，至少这么多近期轮次逐字保留。
    min_recent_rounds: Annotated[int, Field(ge=1, le=10)] = 2
    # 持久化对话的保留上限；先达到者触发清理。
    retention_conversations: PositiveInt = 200
    retention_days: PositiveInt = 180
    # 会话重新加载历史时最多回放多少轮；更早的历史已由 summary_segments
    # 承载，因此不必把整条对话记录读进内存（长对话重载的峰值由此封顶）。
    max_loaded_rounds: PositiveInt = 200
    # 窗口内保留的轮次硬上限：即便 token 未超预算，轮次数超过它也会触发压缩。
    # token 阈值只管住"多大"，管不住"多少轮"，而轮次本身也是内存。
    max_window_rounds: PositiveInt = 40


class EmbeddingSettings(FrozenModel):
    """远程 /embeddings 端点（docs 03.6.4 LTM 语义记忆）。

    与 ProviderSettings 同构：base_url 指向 OpenAI 兼容的 embeddings 端点
    （如智谱 embedding-3、阿里 text-embedding-v3），凭据只从环境变量读取。
    ``base_url`` 留空即关闭嵌入：语义记忆仍可写入，但检索退化为键匹配。
    """

    base_url: str | None = None
    env_key: str | None = None
    model_name: str = ""
    timeout_s: Annotated[float, Field(gt=0)] = 30.0


class VectorDbSettings(FrozenModel):
    """向量库连接（docs 03.6.4：Qdrant）。

    记录本体存 SQLite（memory.db），向量库只存向量 + 点 ID + user_id 载荷；
    服务不可用时检索按 "SQLite BLOB 暴力余弦 → 键匹配" 逐级降级，核心记忆
    链路不依赖它的可用性。``dim`` 留 0 表示由首次嵌入的实际维度推断。
    """

    kind: Literal["qdrant"] = "qdrant"
    url: str | None = None
    collection: str = "finharness_semantics"
    api_key_env: str | None = None
    timeout_s: Annotated[float, Field(gt=0)] = 10.0
    # 大于 0 时用于建集合（并校验嵌入维度一致）；0 = 由首次嵌入推断。
    dim: Annotated[int, Field(ge=0, le=8192)] = 0


class LtmSettings(FrozenModel):
    """跨对话长期记忆（docs 03.6.4）：情节记忆 + 语义记忆。

    情节事件分三类：task_result（任务结果，每轮随结论落库，无 LLM）、
    decision / excerpt（懒蒸馏产出）。语义记忆（事实/概念/偏好）同样由
    蒸馏产出，与情节共用**同一次** LLM 调用，因此开启它不额外增加成本。
    """

    # 对话闲置多久后可被懒蒸馏（与 server.session_ttl_s 同量级）。
    distill_idle_s: PositiveInt = 1800
    # 每轮扫描周期内最多蒸馏多少个对话，使成本有界。
    distill_batch: Annotated[int, Field(ge=1, le=20)] = 3
    # 同一对话蒸馏失败重试上限，超出即放弃（绝不阻塞对话本身）。
    distill_max_attempts: Annotated[int, Field(ge=1, le=10)] = 3
    # 对话首轮被动注入的最近情节数。
    recent_episodes: Annotated[int, Field(ge=0, le=20)] = 5
    # 【跨对话记忆】区块的注入 token 上限。
    inject_max_tokens: PositiveInt = 600
    # 标的驱动的情节召回注入上限。
    recall_max_tokens: PositiveInt = 400
    # LTM 独立保留预算（与源对话的 retention_* 无关：记忆自包含，
    # 源对话被 prune 删除不影响已蒸馏的内容）。
    retention_episodes: PositiveInt = 500
    retention_days: PositiveInt = 365
    # 语义记忆（事实/概念/偏好）是否随蒸馏产出。与情节共用一次 LLM 调用。
    distill_semantics: bool = True
    # 语义保留预算与情节分开：偏好这类条目少而长期，比情节更耐久。
    retention_facts: PositiveInt = 300
    retention_facts_days: PositiveInt = 730
    # 向量召回条数与注入上限（仅在配置了 embeddings 时生效）。
    semantic_top_k: Annotated[int, Field(ge=1, le=20)] = 5
    semantic_inject_max_tokens: PositiveInt = 400
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_db: VectorDbSettings = Field(default_factory=VectorDbSettings)


class AuthSettings(FrozenModel):
    """注册登录（docs 03.13）：会话令牌 TTL、Cookie 属性与端点限速。

    ``secure_cookie`` 默认 False 是因为默认部署绑定 127.0.0.1（HTTP）；
    对外经 HTTPS 暴露的部署应设为 true，使 Cookie 只经加密信道传输。

    限速默认值是"够挡住脚本、不打扰真人"的量级：登录 10 次/5 分钟、注册
    5 次/小时（**按客户端地址**计数）。设为 0 表示关闭该项限制。
    """

    token_ttl_s: PositiveInt = 14 * 24 * 3600
    min_password_len: Annotated[int, Field(ge=8, le=128)] = 8
    secure_cookie: bool = False
    allow_register: bool = True
    # 登录尝试上限（按客户端地址 + 用户名归一化后的键计数）。
    login_max_attempts: NonNegativeInt = 10
    login_window_s: NonNegativeInt = 300
    # 注册上限（按客户端地址计数）。
    register_max_attempts: NonNegativeInt = 5
    register_window_s: NonNegativeInt = 3600
    # 首个注册用户是否自动继承单用户时代的存量数据（对话、笔记、供应商配置）。
    # 默认关闭：这是一个"第一个注册的人拿到全部历史数据"的隐式授权，在对外
    # 部署里等于把存量数据交给任意外部访客。需要时由运维显式打开。
    claim_legacy_on_first_register: bool = False


class QuotaSettings(FrozenModel):
    """每租户用量预算（隔离方案 P0-6）。

    与 ``context.max_turns`` 的分工：那一个是**单次对话**内模型可以走多少轮，
    属于上下文预算；这里限制的是**一个租户在时间窗口内**能发起多少轮对话，
    属于资源公平性。没有它，一个租户就能占满进程、上游数据源与模型配额。

    ``turns_per_window`` 默认宽松（600 轮/小时，约每分钟 10 轮）：它挡的是
    脚本化滥用，而不是正常的高强度使用。窗口滚动，因此**不会被异常路径
    永久性污染**——这一点是它作为默认防线的原因。

    ``max_concurrent_streams`` 默认 **0（不限制）**，理由是它依赖"会话一定会
    被释放"这一前提，而该前提并不总成立：SSE 生成器若未被消费完（客户端在
    第一个事件前就断开），``release`` 走不到，busy 标记会一直留着，直到被
    回收（``server.busy_timeout_s``，默认 1 小时）。若同时开启并发上限，
    几条这样的泄漏就足以让**整个租户**被挡在门外一小时——那是比要防的滥用
    更糟的自我拒绝服务。因此它保留为显式开关：只有在部署已确认异常路径会
    及时释放（或回收窗口足够短）时才打开。
    """

    turns_per_window: NonNegativeInt = 600
    window_s: NonNegativeInt = 3600
    # 同一租户可同时进行的对话流；0 表示不限制（默认，理由见类文档）。
    max_concurrent_streams: NonNegativeInt = 0


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
    # busy 会话超过此时长仍未释放即视为被遗弃，可被回收（默认 2×会话 TTL）。
    # 保证一个卡死的流不会永久占着一份 AgentLoop 与整条记忆。
    busy_timeout_s: PositiveInt = 3600
    # 进程级用户级缓存（每个用户一份 DataAccess）的上限；超出按 LRU 淘汰。
    user_cache_size: PositiveInt = 64
    static_dir: Path = Path("src/finharness/server/static")
    allow_remote: bool = False


class PathSettings(FrozenModel):
    """产物、缓存与**状态**三类路径。

    ``state_dir`` 存放绝不能被 agent 工具触碰的东西：主加密密钥、用户/令牌库、
    对话库与加密配置库。它必须与 ``output_dir``、``data.cache_dir`` 分开——
    后两者是 agent 可达的（``read_file``/``write_file``/``read_pdf`` 与
    ``/v1/artifacts`` 都开放其中的子树），把密钥与租户数据库放在那里等于让
    "读一个缓存文件"与"读到全部租户的供应商 key"之间只隔一次路径检查。
    跨字段约束见 ``Settings.validate``。
    """

    output_dir: Path = Path("output")
    state_dir: Path = Path("state")
    memory_db: Path = Path("state/memory.db")
    # 用户与会话令牌存储（docs 03.13）。
    auth_db: Path = Path("state/users.db")
    # 加密后的供应商配置库与其主密钥（docs 03.13）。
    config_db: Path = Path("state/config.db")
    secret_key: Path = Path("state/secret.key")
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
    ltm: LtmSettings = Field(default_factory=LtmSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    quota: QuotaSettings = Field(default_factory=QuotaSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)

    @field_validator("providers")
    @classmethod
    def freeze_providers(
        cls, value: Mapping[str, ProviderSettings]
    ) -> Mapping[str, ProviderSettings]:
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def validate_state_separation(self) -> "Settings":
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


def _path_within(child: Path, root: Path) -> bool:
    """``child`` 是否等于 ``root`` 或位于其下（两者均解析后再比较）。

    解析失败（非法字符、过长路径等）时返回 False：宁可放过一次可疑布局，
    也不要因为一个解析异常让配置加载整体失败。
    """
    try:
        resolved_child = Path(child).resolve()
        resolved_root = Path(root).resolve()
    except (OSError, ValueError):
        return False
    return resolved_child == resolved_root or resolved_child.is_relative_to(resolved_root)


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
        ("paths", "state_dir"),
        ("paths", "memory_db"),
        ("paths", "auth_db"),
        ("paths", "config_db"),
        ("paths", "secret_key"),
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
    """把 pydantic 校验错误整理为单行可读信息。

    模型级校验（``@model_validator``）抛出的 ``SettingsError`` 会被 pydantic 包成
    ``value_error``，其 `msg` 前缀是无信息量的 "Value error, "；剥掉它，使状态
    布局这类跨字段错误读起来与其它配置错误一致。
    """

    details: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        message = str(error["msg"])
        if message.startswith("Value error, "):
            message = message[len("Value error, "):]
        details.append(f"{location}: {message}" if location else message)
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
    "FINH_TOOLS_RESULT_TOKEN_OVERRIDES": (("tools", "result_token_overrides"), dict[str, int]),
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
    "FINH_CONTEXT_COMPACTION_RESULT_TOKENS": (("context", "compaction_result_tokens"), int),
    "FINH_CONTEXT_MAX_TOOL_SCHEMA_TOKENS": (("context", "max_tool_schema_tokens"), int),
    "FINH_CONTEXT_MAX_IDENTICAL_TOOL_CALLS": (("context", "max_identical_tool_calls"), int),
    "FINH_CONTEXT_PLAN_STALL_TURNS": (("context", "plan_stall_turns"), int),
    "FINH_CONTEXT_SUMMARY_BUDGET_RATIO": (("context", "summary_budget_ratio"), float),
    "FINH_CONTEXT_MIN_RECENT_ROUNDS": (("context", "min_recent_rounds"), int),
    "FINH_CONTEXT_RETENTION_CONVERSATIONS": (("context", "retention_conversations"), int),
    "FINH_CONTEXT_RETENTION_DAYS": (("context", "retention_days"), int),
    "FINH_CONTEXT_SHORT_MEM_CAP": (("context", "short_mem_cap"), int),
    "FINH_CONTEXT_RECALL_MAX_TOKENS": (("context", "recall_max_tokens"), int),
    "FINH_CONTEXT_SUMMARY_INJECT_MAX_TOKENS": (("context", "summary_inject_max_tokens"), int),
    "FINH_LTM_DISTILL_IDLE_S": (("ltm", "distill_idle_s"), int),
    "FINH_LTM_DISTILL_BATCH": (("ltm", "distill_batch"), int),
    "FINH_LTM_DISTILL_MAX_ATTEMPTS": (("ltm", "distill_max_attempts"), int),
    "FINH_LTM_RECENT_EPISODES": (("ltm", "recent_episodes"), int),
    "FINH_LTM_INJECT_MAX_TOKENS": (("ltm", "inject_max_tokens"), int),
    "FINH_LTM_RECALL_MAX_TOKENS": (("ltm", "recall_max_tokens"), int),
    "FINH_LTM_RETENTION_EPISODES": (("ltm", "retention_episodes"), int),
    "FINH_LTM_RETENTION_DAYS": (("ltm", "retention_days"), int),
    "FINH_LTM_DISTILL_SEMANTICS": (("ltm", "distill_semantics"), bool),
    "FINH_LTM_RETENTION_FACTS": (("ltm", "retention_facts"), int),
    "FINH_LTM_RETENTION_FACTS_DAYS": (("ltm", "retention_facts_days"), int),
    "FINH_LTM_SEMANTIC_TOP_K": (("ltm", "semantic_top_k"), int),
    "FINH_LTM_SEMANTIC_INJECT_MAX_TOKENS": (("ltm", "semantic_inject_max_tokens"), int),
    "FINH_LTM_EMBEDDINGS_BASE_URL": (("ltm", "embeddings", "base_url"), str),
    "FINH_LTM_EMBEDDINGS_ENV_KEY": (("ltm", "embeddings", "env_key"), str),
    "FINH_LTM_EMBEDDINGS_MODEL_NAME": (("ltm", "embeddings", "model_name"), str),
    "FINH_LTM_EMBEDDINGS_TIMEOUT_S": (("ltm", "embeddings", "timeout_s"), float),
    "FINH_LTM_VECTOR_DB_KIND": (("ltm", "vector_db", "kind"), Literal["qdrant"]),
    "FINH_LTM_VECTOR_DB_URL": (("ltm", "vector_db", "url"), str),
    "FINH_LTM_VECTOR_DB_COLLECTION": (("ltm", "vector_db", "collection"), str),
    "FINH_LTM_VECTOR_DB_API_KEY_ENV": (("ltm", "vector_db", "api_key_env"), str),
    "FINH_LTM_VECTOR_DB_TIMEOUT_S": (("ltm", "vector_db", "timeout_s"), float),
    "FINH_LTM_VECTOR_DB_DIM": (("ltm", "vector_db", "dim"), int),
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
    "FINH_AUTH_LOGIN_MAX_ATTEMPTS": (("auth", "login_max_attempts"), int),
    "FINH_AUTH_LOGIN_WINDOW_S": (("auth", "login_window_s"), int),
    "FINH_AUTH_REGISTER_MAX_ATTEMPTS": (("auth", "register_max_attempts"), int),
    "FINH_AUTH_REGISTER_WINDOW_S": (("auth", "register_window_s"), int),
    "FINH_AUTH_CLAIM_LEGACY_ON_FIRST_REGISTER": (
        ("auth", "claim_legacy_on_first_register"),
        bool,
    ),
    "FINH_QUOTA_TURNS_PER_WINDOW": (("quota", "turns_per_window"), int),
    "FINH_QUOTA_WINDOW_S": (("quota", "window_s"), int),
    "FINH_QUOTA_MAX_CONCURRENT_STREAMS": (("quota", "max_concurrent_streams"), int),
    "FINH_PATHS_OUTPUT_DIR": (("paths", "output_dir"), Path),
    "FINH_PATHS_STATE_DIR": (("paths", "state_dir"), Path),
    "FINH_PATHS_MEMORY_DB": (("paths", "memory_db"), Path),
    "FINH_PATHS_AUTH_DB": (("paths", "auth_db"), Path),
    "FINH_PATHS_CONFIG_DB": (("paths", "config_db"), Path),
    "FINH_PATHS_SECRET_KEY": (("paths", "secret_key"), Path),
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
