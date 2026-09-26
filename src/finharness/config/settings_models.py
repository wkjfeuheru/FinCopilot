"""配置的嵌套模型：每个配置节的字段、默认值、字段级校验。

这些类只描述"配置长什么样"，不含加载逻辑（读文件、环境变量覆盖、路径解析在
``settings_loading``），也不含跨节校验（在 ``Settings`` 里）。拆出来使编辑某个
配置节的字段时不必滚动查阅加载实现。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SettingsError(ValueError):
    """配置无法安全加载或启动时抛出。"""


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
CacheKind = Literal["quote", "kline", "indicators", "financials", "announcements", "web", "reports", "macro", "industry", "dataset"]
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
    # 单次回复的输出上限。4096 曾不够用：`write_report` 这类工具的参数本身就是
    # 一份完整研报的正文，一旦触顶，工具参数会被截在半句 JSON 上，整轮以
    # "invalid tool arguments" 失败（见 provider.errors.OutputTruncatedError）。
    # 8192 是 deepseek 系列的默认输出上限，给最重的调用留出余量。
    max_tokens: PositiveInt = 8192
    thinking: ThinkingSettings = Field(default_factory=ThinkingSettings)


class ProviderSettings(FrozenModel):
    kind: ProviderKind = "openai_compat"
    base_url: str | None = None
    env_key: str | None = None
    api_version: str | None = None
    first_byte_timeout_s: Annotated[float, Field(gt=0)] = 30.0
    idle_timeout_s: Annotated[float, Field(gt=0)] = 60.0

    @model_validator(mode="after")
    def validate_endpoint_contract(self) -> ProviderSettings:
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
    # 顺序即优先级：第一个不报错的适配器胜出，其余仅在它失败或不支持该语义方法时
    # 才被尝试。同花顺排在 akshare 之前，是因为它提供后者没有的能力（概念指数成分股、
    # 结构化三表字段、特色数据、基金/期货/期权）；把 akshare 放在前面，一旦它返回了
    # 数据，同花顺就永远不会被用到——这正是数据缺口填不上的原因。
    adapter_order: Annotated[tuple[str, ...], Field(min_length=1)] = Field(
        default_factory=lambda: ("fuyao", "akshare", "tushare", "baostock")
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
            # 数据集派发器承载的是异构的长尾端点（热股榜按小时变、交易日历多年不变、
            # 财报按季）。单一 TTL 只能取其中最保守的一个：过短会让稳定数据反复重取，
            # 过长会把盘中榜单缓存到收盘之后。一天与 quote/kline 同档，是两者之间的
            # 折中；需要更细的粒度时由调用方以更窄的参数（如指定交易日）表达。
            "dataset": 1,
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
    # 是否把每轮的结论自动升格为跨对话情节（kind=task_result，无 LLM、每轮随结论落库）。
    #
    # 默认**关闭**：这是唯一"未经约定"的全局记忆写入路径——用户只是正常问一句，其结论
    # 就会在每轮结束时悄悄外溢成全局记忆，并在之后**任何**新对话的首轮被注入。默认只保留
    # 两条经过约定的来源：显式写入（如 remember_preference）与懒蒸馏产出的 decision/excerpt
    # （受 distill_semantics 管辖）。需要旧行为（每轮结论即全局情节）时显式开启。
    # 关闭不影响对话内的结论/摘要/引用——它们仍按对话隔离地保存。
    auto_task_episodes: bool = False
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

    # 会话令牌 TTL（绝对过期）：从登录时刻起算，到期后任意请求返回 401，
    # 前端收到即回到登录页。会话有服务端状态（auth_sessions 表），改短即生效。
    token_ttl_s: PositiveInt = 3600
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
    # 管理员引导开关：开启期间注册的用户 role='admin'，关闭后注册的都是
    # 普通用户。公网部署产生第一个管理员的标准流程：打开本开关（与注册
    # 开关一起）→ 注册管理员账号 → 删除开关变量。环境变量
    # FINH_AUTH_ADMIN_BOOTSTRAP，Railway Variables 面板即可操作。
    admin_bootstrap: bool = False


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


class TraceStoreSettings(FrozenModel):
    """运行轨迹持久化（docs 03.14.4）：自建监控平台的数据面。

    纯标准库 SQLite，无额外依赖。

    ``admin_users`` 已弃用：监控与管理页的权限统一为 ``users.role='admin'``
    （见 ``UserStore.register`` 的 bootstrap_admin）。字段保留只为旧
    settings.json 兼容（不报错），**不再参与任何鉴权判断**。
    """

    enabled: bool = False
    db_path: Path = Path("state/trace.db")
    admin_users: list[str] = Field(default_factory=list)
    # False 时 thought/answer/preview 只存前 2000 字符（仍脱敏）。
    capture_payloads: bool = True
    # 保留天数；<=0 表示永久保留。清理随蒸馏扫描器的同一后台节奏执行。
    retention_days: int = Field(default=90, ge=0)


class ObservabilitySettings(FrozenModel):
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    trace_store: TraceStoreSettings = Field(default_factory=TraceStoreSettings)


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
    # /metrics 的抓取令牌。远程模式下启用 metrics 时必须设置（见 validate）；
    # 本地回环下可留空，此时 /metrics 保持开放以便开发。比较按常量时间进行。
    metrics_token: str | None = None


class ComputeSettings(FrozenModel):
    """主服务向隔离计算 worker 派发任务的契约。"""

    # 空值只允许本机开发、CLI 与 eval；远程多租户服务必须配置该地址。
    remote_worker_url: str | None = None
    hmac_secret_env: str = "FINH_COMPUTE_HMAC_SECRET"
    lease_seconds: PositiveInt = 60
    max_waiting_per_user: PositiveInt = 2
    max_attempts: PositiveInt = 2


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
    # 用量账本（管理员页数据源）：每轮一行的 token/轮次记录。
    usage_db: Path = Path("state/usage.db")
    # 仅主服务可读写：任务队列与待处理输入包，绝不挂给 worker 或 agent。
    compute_jobs_db: Path = Path("state/compute_jobs.db")
    compute_packages_dir: Path = Path("state/compute_packages")
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


class FuyaoSettings(FrozenModel):
    """同花顺（iFinD / Fuyao）金融数据后端（docs 03.5）。

    以 MCP 形式接入：六个服务各自是一个 Streamable HTTP 端点，共用一个 API Key
    （请求头 ``X-api-key``）。配置方式与 Provider/Search 一致——推荐把凭据放在
    环境变量里（``env_key``），也允许在未被跟踪的 ``settings.json`` 内联明文
    ``api_key`` 作为本地便利。

    加载时缺少凭据并非错误：适配器照常构造，真正被调用时才报告"未配置同花顺
    密钥"，编排器把它当作普通的回退原因交给下一个数据源。这样"没配 key"与
    "启动失败"不会混为一谈，未使用同花顺的部署也不会被它拖住。

    ``catalog_ttl_s`` 缓存的是各服务的 ``tools/list``（数据集目录），不是数据本身：
    目录只在发现工具与校验参数时读取，每次调用都重取会为一次取数多付一个往返。
    """

    enabled: bool = True
    kind: Literal["mcp"] = "mcp"
    base_url: str = "https://fuyao.aicubes.cn"
    env_key: str = "HITHINK_FINANCE_API_KEY"
    # 单次上游请求的超时。默认刻意压到 3s（而非通用 HTTP 客户端的 30s）：指标端点
    # 一次只给一个报告期，``fetch_indicators`` 要逐期发多次请求，若每次允许 30s，
    # 一个慢上游就能吃光整个工具预算（30s）而让整条调用被 engine 判 timeout。
    # 短超时使慢请求快速失败，从而落到回退链的 akshare；正常响应远快于 3s。
    timeout_s: Annotated[float, Field(gt=0)] = 3.0
    api_key: str | None = None
    # 与 ``search.proxy`` 同理：``httpx`` 只读环境变量，而 A 股数据源（``requests``）
    # 还遵循操作系统代理，在不显式设置时两者会走不同的出口。留空则回退到系统代理。
    proxy: str | None = None
    catalog_ttl_s: NonNegativeFloat = 300.0

    def resolved_api_key(self) -> str | None:
        """本次进程可用的凭据：内联 ``api_key`` 优先于环境变量。

        与 ``settings.search`` 同一口径（内联 key 让本地配置无需导出任何环境变量
        即可生效）。密钥从不写进日志或工具结果——它只在这里被读取。
        """
        if self.api_key:
            return self.api_key
        if not self.env_key:
            return None
        return os.getenv(self.env_key)


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
