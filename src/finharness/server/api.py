"""FastAPI 应用与 M0 聊天路由。

除 ``/v1/health``、``/v1/ready``、``/v1/auth/*`` 与 ``/metrics`` 外，所有端点都要求
认证，且所有数据访问都按当前用户隔离（docs 03.13）。
"""

import asyncio
import contextlib
import hmac
import os
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from finharness.auth.dependency import _bearer_token, create_require_user
from finharness.auth.ratelimit import RateLimiter
from finharness.auth.store import CurrentUser, UserStore
from finharness.compute.protocol import TaskSigner
from finharness.config.crypto import SecretCipher
from finharness.config.settings import Settings
from finharness.config.store import ConfigStore
from finharness.context.memory.state_store import SqliteAgentStateStore
from finharness.context.memory.store import MemoryStore, _snapshot_resumable
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.adapters.eastmoney_report_adapter import EastmoneyReportAdapter
from finharness.data.adapters.fuyao_adapter import FuyaoMcpAdapter
from finharness.data.adapters.tavily_adapter import TavilyAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.engine.state import public_state_view
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import HookChain
from finharness.observability import build_observer, get_logger, setup_logging
from finharness.observability.usage_store import UsageStore
from finharness.permissions.gate import (
    EGRESS_CATEGORY,
    ConfirmationSpec,
    PermissionGate,
)
from finharness.provider.resolver import NotConfigured, ProviderResolver
from finharness.server.admin_api import create_admin_router
from finharness.server.auth_api import create_auth_router
from finharness.server.compute_api import create_compute_router
from finharness.server.config_api import create_config_router
from finharness.server.confirm import ConfirmBus
from finharness.server.distill_sweeper import (
    distill_user_backlog,
    start_distill_sweeper,
    stop_distill_sweeper,
)
from finharness.server.sessions import SessionBusyError, SessionCompute, SessionRegistry
from finharness.server.sse import HEARTBEAT_S, encode_comment, encode_event
from finharness.tools.registry import ALL_TOOL_CLASSES, ToolRegistry
from finharness.types import StopSignal
from finharness.utils.bounded import BoundedMap
from finharness.workspace import Workspace, WorkspaceViolation

# 单一定义，从 prompts/system.md 加载，使测试与产品发布使用同一份文本
# （文档说明：它管控规划、引用与收敛）。
DEFAULT_SYSTEM_PROMPT = system_prompt()


class ChatRequest(BaseModel):
    # conversation_id 是记忆作用域，也是客户端的持久句柄；可传入它
    # 以恢复会话已过期的对话。
    conversation_id: str | None = None
    # session_id 是活跃的执行窗口；可选，通常省略，
    # 因为服务端会从对话中解析出它。
    session_id: str | None = None
    message: str
    # Explicit recovery: resume=true continues the latest resumable FSM run
    # (or hydrates a legacy TurnCheckpoint into a new run). A normal message
    # abandons any older resumable run before starting fresh.
    resume: bool = False


class RespondRequest(BaseModel):
    request_id: str
    response: str


class StopRequest(BaseModel):
    """请求停止一次在飞的生成（docs 03.3）。

    两个 id 都可选但至少给一个：``session_id`` 精确指向执行窗口，
    ``conversation_id`` 是客户端持久化的句柄（会话过期重建后仍然有效）。
    """

    session_id: str | None = None
    conversation_id: str | None = None


class QueueSink:
    def __init__(self, trace_recorder=None, round_recorder=None):
        import asyncio

        self.queue = asyncio.Queue()
        self.started_at = time.monotonic()
        self.first_token_at: float | None = None
        self.done_at: float | None = None
        self.replay_events: list[dict] = []
        # 可选的 trace 落库旁路（监控平台，docs 03.14.4）：记录非 text_delta
        # 事件；TraceStore 自身吞错，落库失败绝不影响对话流。
        self.trace_recorder = trace_recorder
        # 轮次轨迹的实时落库钩子（可选）：引擎每完成一轮就调用，使进程崩溃
        # 不再丢掉已发生的轮次（旧行为只在 run 结束时批量写）。
        self.round_recorder = round_recorder

    async def emit(self, event):
        """将引擎事件入队，并记录计时与重放所需的元数据。"""
        now = time.monotonic()
        if event.kind == "text_delta" and self.first_token_at is None:
            self.first_token_at = now
        if event.kind == "done":
            self.done_at = now
        if self.trace_recorder is not None and event.kind != "text_delta":
            self.trace_recorder(event)
        if event.kind in {
            "tool_status",
            "tool_activated",
            "context_compacted",
            # 路由注入了什么方法论是本轮结论的依据之一，重新加载的对话应当仍能看到它。
            "context_routed",
            "loop_guard",
            "plan_progress",
            "interactive_request",
            # FSM public phase view — clients rebuild AgentTrace from these.
            "state",
            # ``answer`` 与终态的 ``done`` 也属于失败轮次的重放记录：
            # 重新加载的对话仍须展示本次运行已确立的部分发现。
            "answer",
            "done",
        }:
            self.replay_events.append(
                {"event": _event_name(event.kind), "data": dict(event.data)}
            )
        await self.queue.put(event)

    def record_round(self, round_trace, *, phase=None, revision=None) -> None:
        """引擎每完成一轮调用一次；无 recorder 时 no-op（观测旁路）。"""
        if self.round_recorder is not None:
            self.round_recorder(round_trace, phase=phase, revision=revision)

    def turn_metadata(self) -> dict | None:
        """汇总本轮的重放事件与耗时元数据；轮次未完成时返回 None。"""
        if self.done_at is None:
            return None
        return {
            "events": self.replay_events,
            "first_token_ms": (
                round((self.first_token_at - self.started_at) * 1000)
                if self.first_token_at is not None
                else None
            ),
            "total_duration_ms": round((self.done_at - self.started_at) * 1000),
        }


def create_app(
    provider=None,
    data_access=None,
    settings: Settings | None = None,
    *,
    config_store: ConfigStore | None = None,
    resolver: ProviderResolver | None = None,
    probe_client_factory=None,
    single_tenant: bool = False,
) -> FastAPI:
    """构建 FastAPI 应用，装配缓存、适配器、会话注册表与全部路由。

    ``data_access`` 是**单租户**注入点：给了它，全体用户共用同一个实例，
    每用户的产物与缓存命名空间随之关闭（见 ``data_for``）。因此它必须与
    ``single_tenant=True`` 同时出现，且在远程监听（``server.allow_remote``）
    下被拒绝——多租户部署里"静默共用一个数据面"是隔离失效，而不是配置选项。
    """
    if data_access is not None and not single_tenant:
        raise ValueError(
            "注入 data_access 会关闭每用户产物/缓存隔离，必须显式声明 "
            "single_tenant=True（仅用于测试与单用户本地运行）"
        )
    if data_access is not None and settings is not None and settings.server.allow_remote:
        raise ValueError(
            "server.allow_remote=true（多租户）下不接受 data_access 注入："
            "全体用户共用一份数据面会静默关闭租户隔离"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """服务启动时拉起 LTM 蒸馏扫描器，关闭时取消它。

        扫描器需要事件循环，而模块级 ``create_app()`` 发生在导入期（无循环），
        因此不能在构造函数里直接 create_task。闭包在调用时才解析变量，所以
        这里可以引用下方才赋值/替换的 ``resolver`` 等局部名。
        """
        app.state.ltm_sweeper_task = start_distill_sweeper(
            app_state=app.state,
            provider_resolver=resolver,
            settings=settings,
            observer=observer,
        )
        # trace 保留期清理与蒸馏扫描器同一生命周期：启动即跑、关闭即停。
        trace_cleanup_task = None
        if trace_store is not None:
            trace_cleanup_task = asyncio.create_task(_trace_cleanup_worker())
        # 向量维护（docs 03.6.4 LTM）：先把历史版本写在 SQLite BLOB 里的存量
        # 向量迁入 Qdrant（配了才有动作，失败留下次启动重试），再回填"先积累
        # 了语义条目、之后才配好 embedding"以及"条目被改写导致向量失效"两种
        # 情况。放在启动时做一次，避免这些条目直到下次蒸馏才重新可召回。
        # 失败无妨——语义召回是增强项，键匹配始终在。
        try:
            semantic_index.migrate_local_vectors()
        except Exception:  # noqa: BLE001 - 迁移绝不该阻止服务启动
            pass
        try:
            for user_id in memory_store.ltm_fact_users():
                semantic_index.index_pending(user_id=user_id)
        except Exception:  # noqa: BLE001 - 回填绝不该阻止服务启动
            pass
        compute_cleanup_task = asyncio.create_task(_compute_cleanup_worker())
        try:
            yield
        finally:
            compute_cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await compute_cleanup_task
            if trace_cleanup_task is not None:
                trace_cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await trace_cleanup_task
            await stop_distill_sweeper(getattr(app.state, "ltm_sweeper_task", None))

    application = FastAPI(title="FinHarness", lifespan=lifespan)

    async def _trace_cleanup_worker() -> None:
        """trace 保留期清理：小时级节奏，失败只等下一轮。"""
        while True:
            await asyncio.sleep(3600.0)
            with contextlib.suppress(Exception):
                await asyncio.to_thread(trace_store.cleanup, trace_cfg.retention_days)  # type: ignore[union-attr]
    settings = settings or Settings.from_file()

    # 日志在任何组件之前配置，使启动期的日志本身就带上下文字段。
    setup_logging(
        level=settings.observability.logging.level,
        json_format=settings.observability.logging.json_format,
        path=settings.observability.logging.path,
    )
    # 观测门面：未启用或未安装可选依赖时自动降级为 no-op，绝不影响主流程。
    observer = build_observer(settings)

    # 运行轨迹落库（监控平台数据面，docs 03.14.4）：默认关闭；纯标准库
    # SQLite，无可选依赖。所有写入吞错——trace 缺失不该影响任何一次对话。
    trace_store = None
    trace_cfg = settings.observability.trace_store
    if trace_cfg.enabled:
        from finharness.observability.trace_store import TraceStore

        trace_store = TraceStore(
            trace_cfg.db_path, capture_payloads=trace_cfg.capture_payloads
        )

    # -- 认证（docs 03.13）-----------------------------------------------------
    user_store = UserStore(settings.paths.auth_db)
    application.state.user_store = user_store
    require_user = create_require_user(user_store)

    # 用量账本（管理员页数据源）：每轮一行，永远记录，与 trace 开关解耦。
    usage_store = UsageStore(settings.paths.usage_db)
    application.state.usage_store = usage_store

    # 对话记忆：一个存储服务所有对话；对话记录、
    # 引用、结论与摘要分段均以 conversation id 为键。
    # （提前构造：首个注册用户的存量认领需要它。）
    memory_store = MemoryStore(settings.paths.memory_db)
    application.state.memory_store = memory_store

    # 队列数据库仅由主服务持有；worker 只能经签名内部接口领取任务，绝不能
    # 直接挂载 state 或打开这个 SQLite 文件。
    from finharness.compute.executor import RemoteComputeExecutor
    from finharness.compute.queue import ComputeJobStore

    application.state.compute_jobs = ComputeJobStore(
        settings.paths.compute_jobs_db,
        max_waiting_per_user=int(settings.compute.max_waiting_per_user),
        max_attempts=int(settings.compute.max_attempts),
    )
    # 只有配了远程 worker 地址才启用隔离计算通道（远程模式强制要求该地址，
    # 见 settings.validate）。否则把工具的重活外包给一个不存在的 worker，只会
    # 在本地单机/开发时把每次 docx 导出拖到超时——未配置即回退进程内执行。
    application.state.compute_executor = (
        RemoteComputeExecutor(
            store=application.state.compute_jobs, packages_dir=settings.paths.compute_packages_dir,
            output_dir=settings.paths.output_dir,
        )
        if settings.compute.remote_worker_url
        else None
    )

    async def _compute_cleanup_worker() -> None:
        executor = application.state.compute_executor
        if executor is None:
            return
        while True:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(executor.cleanup_terminal_packages)
            await asyncio.sleep(30)
    compute_secret = os.getenv(settings.compute.hmac_secret_env)
    application.state.compute_signer = (
        TaskSigner(compute_secret) if compute_secret and len(compute_secret.encode("utf-8")) >= 16 else None
    )

    # 内部 worker 接口（docs 03.15）：签名校验、任务包与产物的全部边界都在
    # compute_api 内，见该模块的说明。
    application.include_router(
        create_compute_router(settings=settings, state=application.state)
    )

    # 语义记忆的检索层（docs 03.6.4 LTM）：记录本体在 SQLite，向量只用于
    # 召回。未配 embedding 端点时 index.enabled=False，语义区块整体不出现，
    # 记忆的写入与键匹配检索仍然完全可用。
    from finharness.context.memory.vector import SemanticIndex, build_vector_store
    from finharness.provider.embeddings import build_embedder

    embedder = build_embedder(settings)
    semantic_index = SemanticIndex(
        store=memory_store,
        embedder=embedder,
        # 未配 embedding 时连向量后端都不构建：此时语义召回整体关闭，
        # 记忆的写入与键匹配检索仍然完全可用。
        vector_store=build_vector_store(settings, memory_store) if embedder else None,
    )
    application.state.semantic_index = semantic_index

    def store_factory() -> ConfigStore:
        nonlocal config_store
        if config_store is None:
            config_store = ConfigStore(
                settings.paths.config_db,
                cipher=SecretCipher(settings.paths.secret_key),
            )
        return config_store

    def _claim_legacy(user_id: str) -> None:
        """首个注册用户认领单用户时代的存量数据（对话、笔记、供应商配置）。"""
        memory_store.claim_user(user_id)
        store_factory().claim_user(user_id)

    # 认证端点限速（隔离方案 P0-5）：默认按"够挡脚本、不打扰真人"的量级。
    login_limiter = RateLimiter(
        limit=settings.auth.login_max_attempts,
        window_s=settings.auth.login_window_s,
    )
    register_limiter = RateLimiter(
        limit=settings.auth.register_max_attempts,
        window_s=settings.auth.register_window_s,
    )
    # 每租户的对话轮次预算（隔离方案 P0-6）。与 context.max_turns 的分工：
    # 那一个是单次对话内的上下文预算，这里是一个租户在时间窗口内的资源公平性。
    turn_limiter = RateLimiter(
        limit=settings.quota.turns_per_window,
        window_s=settings.quota.window_s,
    )
    application.state.login_limiter = login_limiter
    application.state.register_limiter = register_limiter
    application.state.turn_limiter = turn_limiter

    application.include_router(
        create_auth_router(
            store=user_store,
            ttl_s=settings.auth.token_ttl_s,
            secure_cookie=settings.auth.secure_cookie,
            allow_register=settings.auth.allow_register,
            # 存量数据继承是一个"谁先注册谁拿到全部历史"的隐式授权，默认关闭；
            # 需要时由运维显式打开（settings.auth.claim_legacy_on_first_register）。
            claim_legacy=(
                _claim_legacy
                if settings.auth.claim_legacy_on_first_register
                else None
            ),
            # 管理员引导：开启期间注册的用户 role='admin'。运维流程见
            # AuthSettings.admin_bootstrap——注册完立即关闭。
            bootstrap_admin=settings.auth.admin_bootstrap,
            login_limiter=login_limiter,
            register_limiter=register_limiter,
        )
    )

    if resolver is None:
        resolver = ProviderResolver(store_factory=store_factory, settings=settings)

    # 管理页（/v1/admin）：require_admin 内含 require_user，未登录 401、
    # 非管理员 403。数据合并 user_store / memory_store / usage_store 三源。
    application.include_router(
        create_admin_router(
            user_store=user_store,
            memory_store=memory_store,
            usage_store=usage_store,
            require_user=require_user,
            settings=settings,
            semantic_index=semantic_index,
        )
    )

    # 一个缓存服务整个进程；每个会话有自己的引用注册表，
    # 使来源信息限定在该对话内。调用方提供的 DataAccess 按原样使用
    # （测试传入隔离的替身），因此这里绝不改动它。
    data_cache = LocalCache(settings.data.cache_dir)
    adapters = [
        # 同花顺排在最前：它提供 akshare 没有的能力（结构化三表字段、完整财务指标、
        # 概念指数成分股、特色数据、基金/期货/期权），而回退是有序的——放在后面就
        # 意味着 akshare 一旦答得上，它就永远不会被用到。启动时缺少密钥无妨：工具
        # 被调用时才报告"未配置"，编排器把它当作普通的回退原因交给 akshare。
        FuyaoMcpAdapter(
            api_key=settings.fuyao.resolved_api_key(),
            base_url=settings.fuyao.base_url,
            timeout_s=settings.fuyao.timeout_s,
            proxy=settings.fuyao.proxy,
            throttle_seconds=settings.data.throttle_seconds,
        ),
        AkShareAdapter(throttle_seconds=settings.data.throttle_seconds),
        # Web 访问走同一条适配器链与缓存；启动时缺少 key 无妨——
        # 工具在被调用时会报告"未配置"。settings.json 中的内联 key
        # 优先于环境变量，因此本地配置的 key 无需导出任何东西即可生效。
        TavilyAdapter(
            api_key=settings.search.api_key
            or (os.getenv(settings.search.env_key) if settings.search.env_key else None),
            base_url=settings.search.base_url,
            timeout_s=settings.search.timeout_s,
            proxy=settings.search.proxy,
        ),
        # 研报无需 key。``local_pdf_fallback`` 只控制其可选的
        # 全文，全文会从本机访问文档 CDN。
        EastmoneyReportAdapter(
            timeout_s=settings.search.timeout_s,
            with_text_allowed=settings.search.local_pdf_fallback,
            pdf_dir=Path(settings.data.cache_dir) / "pdf",
        ),
    ]

    def adapters_for(user_settings: Settings) -> list:
        """某用户数据面所用的适配器。

        取数适配器（同花顺/akshare/tushare）与检索适配器全局共享一份：其节流状态与
        （同花顺的）MCP 会话保护的都是上游数据源，按用户复制会把上游请求速率乘以
        用户数，还会让每个用户各握一个会话。唯一必须按用户隔离的是研报适配器——
        它的全文落盘目录随后要由该用户的 ``read_pdf`` 读回。
        """
        return [
            adapter.with_pdf_dir(Path(user_settings.data.cache_dir) / "pdf")
            if isinstance(adapter, EastmoneyReportAdapter)
            else adapter
            for adapter in adapters
        ]

    shared_data = data_access or DataAccess(
        adapters,
        cache=data_cache,
        settings=settings,
    )

    # 每个用户的工具级 DataAccess：与 shared_data 共享适配器，但 output/ 换成
    # output/<user>/、缓存换成 users/<user>/，因此用户之间互不可见——包括缓存载荷。
    # 测试注入的 DataAccess 替身按原样使用，不引入用户目录。
    # 有界：一个进程可能见过很多用户，缓存按 LRU 封顶，避免只增不减。
    user_data_access: BoundedMap[str, DataAccess] = BoundedMap(
        max_size=settings.server.user_cache_size
    )

    def scoped_settings(user_id: str) -> Settings:
        """该用户的 settings：产物、缓存与记忆一并落到自己的命名空间下。

        缓存目录必须一起换掉。只换 ``output_dir`` 时，parquet 载荷仍在全局
        ``data_cache/parquet`` 下，而 ``read_file``/``read_pdf``/``/v1/artifacts``
        都开放该子树——缓存文件名又是由 endpoint+params 决定的哈希，于是任何
        知道请求形状的租户都能算出他人的载荷路径。目录分开后，"读不到"由
        包含性检查强制，而不是靠猜不到文件名。
        """
        return settings.model_copy(
            update={
                "paths": settings.paths.model_copy(
                    update={"output_dir": settings.paths.output_dir / user_id}
                ),
                "data": settings.data.model_copy(
                    update={"cache_dir": settings.data.cache_dir / "users" / user_id}
                ),
            }
        )

    def data_for(user_id: str) -> DataAccess:
        if data_access is not None:
            return shared_data
        scoped = user_data_access.get(user_id)
        if scoped is None:
            user_settings = scoped_settings(user_id)
            scoped = DataAccess(
                adapters_for(user_settings),
                cache=LocalCache(user_settings.data.cache_dir),
                settings=user_settings,
            )
            user_data_access[user_id] = scoped
        return scoped

    # 暴露出来，使测试能直接断言一个用户的命名空间边界（产物与缓存同源派生），
    # 而不必通过工具间接推断。
    application.state.data_for = data_for
    application.state.scoped_settings = scoped_settings

    # 治理：一个确认总线桥接所有会话的交互式请求。
    confirm_bus = ConfirmBus(ttl_s=settings.server.confirm_ttl_s)
    # 暴露出来，使测试与运维人员可以检查待处理的交互式请求。
    application.state.confirm_bus = confirm_bus
    # 引用跟随对话，因此产生它们的执行会话过期后仍可寻址。会话级的引用与
    # 上下文**不**在这里另存一份：它们属于 session.loop（见 loop_factory），
    # 由 SessionRegistry 的 TTL 统一回收。否则会形成"会话状态两个所有者"，
    # 而注册表淘汰够不到这里的副本，导致每个会话都永久残留一份记忆。
    # 有界：一个进程见过的对话数可能远超活跃数，按 LRU 封顶。
    conversation_citations: BoundedMap[str, CitationRegistry] = BoundedMap(
        max_size=settings.context.retention_conversations
    )
    # 暴露出来，使测试与运维人员可以检查该缓存的大小（有界性是刻意设计）。
    application.state.conversation_citations = conversation_citations
    fallback_provider = provider

    memory_store.prune(
        max_conversations=settings.context.retention_conversations,
        max_age_days=settings.context.retention_days,
    )
    # 跨对话长期记忆的独立保留（docs 03.6.4 LTM）：与源对话预算无关——
    # 记忆自包含，源对话被删不影响它；这条兜底覆盖"只写 task_result、
    # 从不蒸馏"的用户，避免其 LTM 无界增长。
    memory_store.prune_all_ltm_episodes(
        max_episodes=settings.ltm.retention_episodes,
        max_age_days=settings.ltm.retention_days,
    )
    memory_store.prune_all_ltm_facts(
        max_facts=settings.ltm.retention_facts,
        max_age_days=settings.ltm.retention_facts_days,
    )
    # 对话级的"已确认风险类别"（docs 03.7.1）：网络外发首次确认后在本对话
    # 免问。与 conversation_citations 同生命周期——对话结束、注册表淘汰时
    # 一并消亡，因此免问授权不会活得比它所授权的对话更久。
    confirmed_categories: BoundedMap[str, set[str]] = BoundedMap(
        max_size=settings.context.retention_conversations
    )

    def loop_factory(
        session_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str = "",
    ) -> AgentLoop:
        """为一次会话构建 AgentLoop，并接线引用、上下文、权限门与审计钩子。"""
        if fallback_provider is not None:
            selected = fallback_provider
        else:
            selected = resolver.current(user_id)
        citations = CitationRegistry()
        ctx = ResearchContext(cite=citations, settings=settings)
        # 会话级状态只挂在 loop 上（cite/ctx 都会传进 AgentLoop），由
        # SessionRegistry 的 TTL 连同 session 一起回收；这里只额外登记
        # 对话级引用，因为对话比执行窗口长寿。
        if conversation_id:
            conversation_citations[conversation_id] = citations

        # 工具 schema 按会话生成，因为惰性激活是会话作用域的；
        # ctx 与 registry 相互接线。产物经由 data_for(user_id) 写入
        # output/<user>/，因此用户之间互不可见。
        session_data = data_for(user_id)
        registry_for_session = ToolRegistry(session_data, ctx=ctx, settings=settings)
        audit = AuditHook(
            AuditLogWriter(settings.audit.log_path),
            session_id=session_id or "local",
            user_id=user_id,
        )
        # 闸门必须用**该用户自己的** settings：白名单根取自 settings，若用全局
        # settings，``output/<他人>/`` 与整棵 data_cache/ 都会被判成"产物写入、
        # 免确认"，而工具侧随后又按用户根拒绝——确认决策就建立在比实际操作更
        # 宽的根上。确认对话作用于对话（而非会话）：会话过期重建后，用户在本
        # 对话中给出的"不再询问"仍然有效。
        confirmed = confirmed_categories.get(conversation_id or "local")
        if confirmed is None:
            confirmed = set()
            confirmed_categories[conversation_id or "local"] = confirmed

        # 会话级"始终允许此类操作"：loop_factory 每个执行会话只被调用一次，
        # 因此这份集合天然随会话生灭——新建会话、或会话被 TTL 淘汰后重建，
        # 都会拿到一份空集，"始终允许"不会越过会话边界（与对话级的
        # confirmed_categories 不同，后者在会话重建后仍有效）。
        # y_session 由 PermissionGate.resolve 写入本集合；确认交互走
        # InteractivePort，不再注入 confirm 回调。
        session_approved: set[str] = set()

        gate = PermissionGate(
            settings=getattr(session_data, "settings", None) or settings,
            conversation_id=conversation_id or "local",
            confirmed_categories=confirmed,
            session_approved=session_approved,
        )
        loop = AgentLoop(
            provider=selected,
            registry=registry_for_session,
            settings=settings,
            system=DEFAULT_SYSTEM_PROMPT,
            cite=citations,
            session_id=session_id,
            ctx=ctx,
            gate=gate,
            hooks=HookChain([audit]),
            conversation_id=conversation_id,
            store=memory_store,
            user_id=user_id,
            observer=observer,
            semantic_index=semantic_index,
            # 会话绑定的隔离计算通道（docs 03.15）：user/conversation 到此才
            # 可知，工具不该自己拼；声明 needs_compute 的工具由循环注入。
            # 未配远程 worker 时传 None（而非空壳通道），工具据此回退进程内。
            compute=(
                SessionCompute(
                    application.state.compute_executor, user_id, conversation_id or "local"
                )
                if application.state.compute_executor is not None
                else None
            ),
        )

        class _ConfirmBusPort:
            """InteractivePort：ConfirmBus 持有 ephemeral request_id。"""

            async def prompt(self, spec: ConfirmationSpec) -> str | None:
                kind = "question" if spec.kind == "question" else "confirm"
                dedupe_key = None
                if spec.category == EGRESS_CATEGORY:
                    dedupe_key = (
                        f"egress:{session_id or 'local'}:{conversation_id or 'local'}"
                    )
                payload, answer = await confirm_bus.request(
                    session_id=session_id or "local",
                    kind=kind,
                    prompt=spec.prompt,
                    options=list(spec.options),
                    multi_select=spec.multi_select,
                    user_id=user_id,
                    announce=lambda item: loop._emit("interactive_request", item),
                    dedupe_key=dedupe_key,
                )
                # Do not emit interaction_resolved here: ConfirmationResolved must
                # persist first. Loop emits from _after_dispatch after that revision.
                loop.queue_interaction_resolved(
                    {
                        "request_id": payload.get("request_id"),
                        "answer": answer,
                        "timeout": answer is None,
                    }
                )
                return answer

        # 构造之后注入，使回调可以通过此 loop 发送事件。
        loop.interactive = _ConfirmBusPort()
        loop.audit = audit
        # LTM 懒蒸馏的兜底翼（docs 03.6.4）：用户开新对话时补蒸馏其闲置的
        # 未处理对话。SSE 断线与 TTL 回收都跳不出这条路径——只要用户还在
        # 用，漏网的蒸馏最终都会在这里补上。fire-and-forget：绝不阻塞新对话。
        if conversation_id:
            asyncio.create_task(
                distill_user_backlog(
                    provider=selected,
                    store=memory_store,
                    settings=settings,
                    user_id=user_id,
                    exclude_conversation=conversation_id,
                    observer=observer,
                    index=semantic_index,
                )
            )
        return loop

    registry = SessionRegistry(
        loop_factory,
        ttl_s=settings.server.session_ttl_s,
        busy_timeout_s=settings.server.busy_timeout_s,
        compute_executor=application.state.compute_executor,
    )
    application.state.session_registry = registry
    # LTM 懒蒸馏的前一翼（docs 03.6.4）：周期扫描闲置对话并补蒸馏。
    # 文档曾承诺的"每 60s 会话 TTL 回收"并不存在（注册表是惰性淘汰且无回调），
    # 因此蒸馏不等会话结束——闲置判定直接读 conversations.last_active_at。
    # 实际启动在 lifespan 中（需要事件循环）。
    application.include_router(
        create_config_router(
            store_factory=store_factory,
            resolver=resolver,
            settings=settings,
            probe_client_factory=probe_client_factory,
            require_user=require_user,
        )
    )
    # 监控查询 API（docs 03.14.4）：始终挂载——未启用时数据端点返回 503
    # （带开启方法）而非 404，否则运维方只看到一个无信息量的"加载失败"。
    # 路由内部按 admin_users 白名单二次鉴权。
    from finharness.server.trace_api import create_trace_router

    application.include_router(
        create_trace_router(trace_store, require_user=require_user)
    )
    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if (frontend_dist / "assets").is_dir():
        application.mount(
            "/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets"
        )

    @application.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/ready")
    async def ready() -> JSONResponse:
        def check_audit_log_parent() -> None:
            audit_parent = settings.audit.log_path.parent
            if (
                not audit_parent.exists()
                or not audit_parent.is_dir()
                or not os.access(audit_parent, os.W_OK)
            ):
                raise PermissionError("audit log parent is not writable")

        checks = (
            ("user_store", user_store.ping),
            ("memory_store", memory_store.ping),
            ("config_store", lambda: store_factory().ping()),
            ("usage_store", usage_store.ping),
            ("audit_log_parent", check_audit_log_parent),
        )
        for dependency, check in checks:
            try:
                check()
            except Exception as exc:  # noqa: BLE001 - readiness 必须把依赖异常映射为 503
                get_logger("finharness.server.api").warning(
                    "readiness_check_failed",
                    extra={
                        "dependency": dependency,
                        "error_type": type(exc).__name__,
                    },
                )
                return JSONResponse(
                    status_code=503,
                    content={"status": "not_ready"},
                )
        return JSONResponse(content={"status": "ready"})

    metrics_recorder = getattr(observer, "metrics", None)
    if metrics_recorder is not None:

        @application.get("/metrics")
        async def metrics_endpoint(request: Request) -> Response:
            """Prometheus 抓取端点；仅在 metrics 启用时注册。

            配了 ``server.metrics_token`` 时按常量时间比对 Bearer 令牌——远程
            部署强制要求该令牌（见 Settings.validate），避免公网无鉴权读取运营
            指标。本地回环开发可不设，此时保持开放。
            """
            expected = settings.server.metrics_token
            if expected:
                if not hmac.compare_digest(_bearer_token(request), expected):
                    raise HTTPException(
                        status_code=401,
                        detail="metrics 令牌无效",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            return Response(
                content=metrics_recorder.render(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    @application.get("/v1/tools")
    async def tools(
        session_id: str | None = None, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """列出工具目录；给定 session id 时报告其激活状态。"""
        if session_id and session_id in registry.sessions:
            session = registry.sessions[session_id]
            # 会话句柄是全局的：他人的 session_id 视同不存在。
            if session.user_id != user.id:
                raise HTTPException(status_code=404, detail="会话或对话不存在")
            reg = session.loop.registry
            return {
                "tools": reg.names(),
                "resident": reg.resident_names(),
                "lazy": reg.lazy_names(),
                "active": [name for name in reg.names() if reg.is_active(name)],
            }
        return {"tools": [cls.name for cls in ALL_TOOL_CLASSES]}

    @application.post("/v1/chat/respond")
    async def chat_respond(
        body: RespondRequest, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """应答一个待处理的交互式请求（写确认 / ask_user）。"""
        # 只挂起它的用户可以应答；他人的 request_id 视同不存在。
        ok = confirm_bus.respond(
            request_id=body.request_id, value=body.response, user_id=user.id
        )
        if not ok:
            raise HTTPException(status_code=404, detail="请求不存在或已超时")
        return {"ok": True}

    @application.post("/v1/chat/stop")
    async def chat_stop(
        body: StopRequest, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """请求停止当前在飞的生成（docs 03.3）。

        这是**协作式**停止：置位信号后引擎在下一个等待点（流式 chunk、轮次
        边界、工具返回后）收尾。它因此不会丢弃已有成果——本轮结论照常落库，
        断点记为可继续。前端若在宽限期内没等到终止帧，可以再断开连接兜底。

        幂等：停止一个已经结束、或本就不在运行的生成不是错误，返回
        ``{"stopping": false}`` 即可，客户端无需区分这两种情况。
        """
        session = registry.find(
            session_id=body.session_id or None,
            conversation_id=body.conversation_id or None,
        )
        # 会话句柄是全局的：他人的 session/conversation 视同不存在，
        # 与其余按 id 取用的端点同一口径（404 而非 403，不泄露存在性）。
        if session is None or (user.id and session.user_id != user.id):
            raise HTTPException(status_code=404, detail="会话或对话不存在")
        if not session.busy or session.stop_signal is None:
            # 生成已结束或尚未开始：对它而言停止已经成立。
            return {"stopping": False, "conversation_id": session.conversation_id}
        session.stop_signal.request()
        # 引擎若正停在写确认/提问的 future 上，仅置位信号它不会被观察到，
        # 要等 confirm_ttl_s（默认 120s）超时。这里同步解除该等待，与断线
        # 路径（chat_stream 的 CancelledError 分支）做法一致。
        confirm_bus.cancel_session(session.session_id)
        # 审计留痕：谁在哪个对话上停止了生成。失败不影响停止本身。
        audit = getattr(session.loop, "audit", None)
        if audit is not None:
            rounds = int(getattr(session.loop, "rounds", 0) or 0)
            try:
                audit.generation_stopped(
                    conversation_id=session.conversation_id, rounds=rounds
                )
            except Exception:  # noqa: BLE001 - 审计写失败不该阻断停止
                get_logger("finharness.server.api").exception("stop_audit_failed")
        get_logger("finharness.server.api").info(
            "chat_stream_stop_requested",
            extra={
                "session_id": session.session_id,
                "conversation_id": session.conversation_id,
            },
        )
        return {"stopping": True, "conversation_id": session.conversation_id}

    @application.get("/v1/cache/stats")
    async def cache_stats(user: CurrentUser = Depends(require_user)) -> dict:
        """返回**当前用户**数据缓存的命中统计。

        缓存已按用户命名空间分开，因此这里的数字天然只覆盖调用者自己的载荷；
        没有跨租户的聚合视图。
        """
        user_cache = (
            data_for(user.id).cache
            if data_access is None
            else data_cache
        )
        snapshot = (user_cache or data_cache).stats()
        return {
            "entries": snapshot.entries,
            "hits": snapshot.hits,
            "misses": snapshot.misses,
            "hit_ratio": snapshot.hit_ratio,
        }

    @application.get("/v1/memory")
    async def memory_view(
        conversation_id: str | None = None,
        limit: int = 20,
        subject: str | None = None,
        kind: str | None = None,
        user: CurrentUser = Depends(require_user),
    ) -> dict:
        """累积记忆的只读视图。

        取代了设计曾提出的 MEMORY.md 文件视图：Web 层正是人类阅读
        它的地方，而且它保持结构化，而不是每轮都重写到一个文件。
        长期记忆（docs 03.6.4）分两类返回：``episodes``（情节：做过什么）
        与 ``facts``（语义：知道什么，含 kind='preference' 的偏好），
        各自带溯源字段，可按标的/类型过滤。``semantic_search`` 报告当前
        部署是否具备向量召回（配了 embedding 端点为 true）。
        """
        return {
            "notes": memory_store.get_notes(user_id=user.id),
            "facts": [
                {
                    "fa_uid": item.fa_uid,
                    "key": item.key,
                    "kind": item.kind,
                    "statement": item.statement,
                    "subject": item.subject,
                    "confidence": item.confidence,
                    "source_conversation_id": item.source_conversation_id,
                    "source_ts": item.source_ts,
                    "updated_at": item.updated_at,
                    "embedded": item.has_embedding,
                }
                for item in memory_store.list_ltm_facts(
                    user_id=user.id, kind=kind or None, subject=subject, limit=limit
                )
            ],
            "semantic_search": semantic_index.enabled,
            "episodes": [
                {
                    "ep_uid": item.ep_uid,
                    "kind": item.kind,
                    "subject": item.subject,
                    "summary": item.summary,
                    "cids": list(item.cids),
                    "source_conversation_id": item.source_conversation_id,
                    "source_title": item.source_title,
                    "source_ts": item.source_ts,
                    "created_at": item.created_at,
                }
                for item in memory_store.list_ltm_episodes(
                    user_id=user.id,
                    subject=subject,
                    kind=kind,
                    limit=limit,
                )
            ],
            "conversations": [
                {
                    "conversation_id": record.conversation_id,
                    "title": record.title,
                    "created_at": record.created_at,
                    "last_active_at": record.last_active_at,
                }
                for record in memory_store.list_conversations(
                    user_id=user.id, limit=limit
                )
            ],
            "conclusions": (
                [
                    {
                        "subject": item.subject,
                        "text": item.text,
                        "cids": list(item.cids),
                        "ts": item.ts,
                    }
                    for item in memory_store.load_conclusions(
                        conversation_id, limit=limit
                    )
                ]
                if conversation_id
                # 归属不符视同不存在。
                and memory_store.get_conversation(conversation_id, user_id=user.id)
                is not None
                else []
            ),
        }

    class MemoryEpisodePatch(BaseModel):
        kind: str | None = None
        subject: str | None = None
        summary: str | None = None

    @application.patch("/v1/memory/episodes/{ep_uid}")
    async def memory_episode_patch(
        ep_uid: str, body: MemoryEpisodePatch, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """编辑一条跨对话情节（治理：用户发现记忆有误或表述不当）。"""
        try:
            updated = await asyncio.to_thread(
                memory_store.update_ltm_episode,
                ep_uid,
                user_id=user.id,
                kind=body.kind,
                subject=body.subject,
                summary=body.summary,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409, detail="编辑后的情节与已有记忆重复"
            ) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="记忆条目不存在")
        return {
            "ok": True,
            "episode": {
                "ep_uid": updated.ep_uid,
                "kind": updated.kind,
                "subject": updated.subject,
                "summary": updated.summary,
            },
        }

    @application.delete("/v1/memory/episodes/{ep_uid}")
    async def memory_episode_delete(
        ep_uid: str, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """删除一条跨对话情节（治理：遗忘权）。"""
        deleted = await asyncio.to_thread(memory_store.delete_ltm_episode, ep_uid, user_id=user.id)
        if not deleted:
            raise HTTPException(status_code=404, detail="记忆条目不存在")
        return {"ok": True, "ep_uid": ep_uid}

    class MemoryFactPatch(BaseModel):
        kind: str | None = None
        subject: str | None = None
        statement: str | None = None

    @application.patch("/v1/memory/facts/{fa_uid}")
    async def memory_fact_patch(
        fa_uid: str, body: MemoryFactPatch, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """编辑一条语义记忆（事实/概念/偏好）。

        用户在此改写偏好即"显式声明"，与蒸馏口径冲突时直接覆盖——
        ``(user_id, key)`` 的 UPSERT 语义保证只有一个版本的真相。
        """
        try:
            updated = await asyncio.to_thread(
                memory_store.update_ltm_fact,
                fa_uid,
                user_id=user.id,
                statement=body.statement,
                kind=body.kind,
                subject=body.subject,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="记忆条目不存在")
        # 表述变了旧向量即失效；有索引时立即重算。
        if semantic_index.enabled:
            try:
                semantic_index.index_fact(user_id=user.id, key=updated.key)
            except Exception:  # noqa: BLE001 - 向量重算是增强项
                pass
        return {
            "ok": True,
            "fact": {
                "fa_uid": updated.fa_uid,
                "key": updated.key,
                "kind": updated.kind,
                "statement": updated.statement,
                "subject": updated.subject,
            },
        }

    @application.delete("/v1/memory/facts/{fa_uid}")
    async def memory_fact_delete(
        fa_uid: str, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """删除一条语义记忆（治理：遗忘权）。"""
        fact = memory_store.get_ltm_fact(fa_uid, user_id=user.id)
        if fact is None or not await asyncio.to_thread(memory_store.delete_ltm_fact, fa_uid, user_id=user.id):
            raise HTTPException(status_code=404, detail="记忆条目不存在")
        # 同步清理向量库中的点，否则后续召回会命中一个已删除的 id。
        semantic_index.unindex_fact(fact, user_id=user.id)
        return {"ok": True, "fa_uid": fa_uid}

    @application.get("/v1/artifacts")
    async def download_artifact(
        path: str, user: CurrentUser = Depends(require_user)
    ):
        """提供产出的文件，范围限于该用户自己的 workspace。

        可达性来自该用户的 ``Workspace``，与 ``read_file``/``read_pdf`` 同源；
        没有校验该端就会变成任意文件读取。缓存侧只开放该用户自己的命名空间
        ——整个 data_cache/ 会连带暴露 users.db（密码哈希）与 memory.db
        （全部用户的对话），而共享一份 parquet/ 又会让任何用户读到他人缓存的
        载荷（文件名是 endpoint+params 的哈希）。
        """
        user_workspace = Workspace(scoped_settings(user.id))
        try:
            target = user_workspace.resolve_read(path)
        except WorkspaceViolation as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(target, filename=target.name)

    @application.get("/v1/citations")
    async def citations(
        session_id: str | None = None,
        conversation_id: str | None = None,
        user: CurrentUser = Depends(require_user),
    ) -> dict:
        """某对话（优先）或某活跃会话的引用。"""
        if conversation_id:
            # 无论走内存还是落库，都先确认对话属于当前用户。
            if memory_store.get_conversation(conversation_id, user_id=user.id) is None:
                raise HTTPException(status_code=404, detail="会话或对话不存在")
            # 名字不能叫 registry：那会遮蔽函数外的 SessionRegistry，
            # 使第 499/507 行的 `registry` 变成未绑定的局部变量。
            citation_registry = conversation_citations.get(conversation_id)
            if citation_registry is not None:
                records = citation_registry.all()
            else:
                # 对话比服务它的进程更长寿，因此回退到持久化的引用：
                # 恢复的对话即使在服务端重启后仍能显示其来源。
                records = memory_store.load_citations(conversation_id, user_id=user.id)
        elif session_id:
            session = registry.sessions.get(session_id)
            if session is None or session.user_id != user.id:
                raise HTTPException(status_code=404, detail="会话或对话不存在")
            records = session.loop.cite.all()
        else:
            # 无过滤参数：只汇总该用户各会话的引用。
            records = [
                item
                for s in registry.sessions.values()
                if s.user_id == user.id
                for item in s.loop.cite.all()
            ]
        items: list[dict] = [
            {
                "cid": item.cid,
                "tool": item.tool,
                "endpoint": item.endpoint,
                "symbol": item.symbol,
                "params": dict(item.params),
                "rows": item.rows,
                "cols": item.cols,
                "from_cache": item.from_cache,
                "ts": item.ts,
                "fingerprint": item.fingerprint,
            }
            for item in records
        ]
        return {"citations": items, "count": len(items)}

    @application.get("/v1/conversations")
    async def conversations(
        limit: int = 50, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """列出已存储的对话，最近活动在前，供选择器使用。"""
        return {
            "conversations": [
                {
                    "conversation_id": record.conversation_id,
                    "title": record.title,
                    "created_at": record.created_at,
                    "last_active_at": record.last_active_at,
                }
                for record in memory_store.list_conversations(
                    user_id=user.id, limit=limit
                )
            ]
        }

    @application.get("/v1/conversations/{conversation_id}/messages")
    async def conversation_messages(
        conversation_id: str,
        limit: int = 200,
        user: CurrentUser = Depends(require_user),
    ) -> dict:
        """重放对话记录，使客户端可以恢复其视图。

        只返回用户轮次与最终回答：中间的工具帧属于工作状态，
        不是读者应该滚动浏览的内容。
        """
        # 先校验归属，再读取：不属当前用户的对话视同不存在。
        if memory_store.get_conversation(conversation_id, user_id=user.id) is None:
            raise HTTPException(status_code=404, detail="对话不存在或无消息")
        messages = memory_store.load_messages(conversation_id)
        if not messages:
            raise HTTPException(status_code=404, detail="对话不存在或无消息")
        rendered: list[dict] = []
        for message in messages:
            if message.role == "user" and message.content:
                rendered.append({"role": "user", "text": message.content})
            elif message.role == "assistant" and message.content:
                item = {"role": "assistant", "text": message.content}
                if isinstance(message.metadata.get("turn"), dict):
                    item["turn"] = message.metadata["turn"]
                rendered.append(item)
        # 上一轮被停止时报告可继续（docs 03.3）：客户端据此在刷新后仍能显示
        # "继续研究"入口，而不是把一次中断当成一次普通的失败。
        # FSM snapshot is authoritative when any row exists; TurnCheckpoint is
        # the legacy fallback only when this conversation has never been snapshotted.
        state_store = SqliteAgentStateStore(memory_store)
        snapshot = state_store.latest(conversation_id, user.id)
        checkpoint = memory_store.load_latest_checkpoint(conversation_id)
        resumable = None
        if snapshot is not None:
            if _snapshot_resumable(snapshot):
                outcome = snapshot.outcome
                resumable = {
                    "reason": (outcome.reason if outcome is not None else None) or "",
                    "rounds": snapshot.turn,
                    "updated_at": snapshot.updated_at,
                    "state": public_state_view(snapshot),
                }
        elif checkpoint is not None and checkpoint.recoverable:
            resumable = {
                "reason": checkpoint.reason,
                "rounds": checkpoint.rounds,
                "plan": checkpoint.plan,
                "updated_at": checkpoint.updated_at,
            }
        return {
            "conversation_id": conversation_id,
            "messages": rendered[-max(limit, 1) :],
            "resumable": resumable,
        }

    @application.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str, user: CurrentUser = Depends(require_user)
    ) -> dict:
        """删除一个对话及其作用域内的所有内容。

        移除对话记录、摘要、引用、结论与标的池。
        用户偏好是用户作用域的，刻意不受影响。

        当对话正处于请求处理中时拒绝：在运行中的 loop 底下删掉存储
        会让它持久化到空处。
        """
        if memory_store.get_conversation(conversation_id, user_id=user.id) is None:
            raise HTTPException(status_code=404, detail="对话不存在")
        active = registry.find_by_conversation(conversation_id)
        if active is not None and active.busy:
            raise HTTPException(
                status_code=409, detail="该对话正在处理中，请稍后再删除"
            )
        await asyncio.to_thread(memory_store.delete_conversation, conversation_id)
        conversation_citations.pop(conversation_id, None)
        # 会话级状态无需在这里清理：它挂在 session/loop 上，随注册表淘汰回收。
        for session_id, session in list(registry.sessions.items()):
            if session.conversation_id == conversation_id:
                registry.sessions.pop(session_id, None)
        return {"ok": True, "conversation_id": conversation_id}

    @application.get("/v1/account/export")
    async def export_account(user: CurrentUser = Depends(require_user)) -> Response:
        """导出自有账号的全部数据（可携带权，隔离方案 Phase 2）。

        与管理员抹除相对：用户可自助取回自己的数据，但删除需管理员。归档含
        ``userdata.json``（记忆库结构）与该用户的产物、缓存文件。
        """
        from finharness.server.tenant_data import export_tenant_archive

        archive = export_tenant_archive(
            store=memory_store, user_id=user.id, settings=settings
        )
        return Response(
            content=archive,
            media_type="application/zip",
            headers={
                "content-disposition": 'attachment; filename="finharness-export.zip"'
            },
        )

    @application.get("/")
    async def root() -> HTMLResponse:
        index = frontend_dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse("<html><body><div id='root'>FinHarness</div></body></html>")

    @application.post("/v1/report")
    async def report_stream(
        request: ChatRequest, user: CurrentUser = Depends(require_user)
    ):
        """将对话转化为一份报告。

        该 endpoint 不添加自己的固定流水线：它在会话中提交一个普通
        请求，因此模型会加载匹配的场景 skill、读取其报告模板并调用
        write_report，与用户亲自输入时完全一样。产出的文件随后
        以 tool_status 附件的形式到达。
        """
        prompt = (
            request.message.strip()
            or "请基于本次会话的研究内容生成一份研报，先加载对应场景技能并参考其报告模板再成稿。"
        )
        return await chat_stream(
            ChatRequest(
                session_id=request.session_id,
                conversation_id=request.conversation_id,
                message=prompt,
            ),
            user=user,
        )

    @application.post("/v1/chat/stream")
    async def chat_stream(
        request: ChatRequest, user: CurrentUser = Depends(require_user)
    ):
        """运行一轮对话并以 SSE 流式返回引擎事件。"""
        from fastapi import HTTPException

        # 归属校验：对话已存在但属于他人时视同不存在（404 而非 403，
        # 不泄露存在性）。完全未知的 id 则按新对话开始——客户端可能
        # 回传一个已被删除的 id，那应当从头开始，而不是报错。
        if request.conversation_id:
            existing = memory_store.get_conversation(request.conversation_id)
            if existing is not None and existing.user_id != user.id:
                raise HTTPException(status_code=404, detail="对话不存在")
        try:
            session = await registry.ensure(
                request.session_id,
                conversation_id=request.conversation_id,
                user_id=user.id,
            )
        except SessionBusyError as exc:
            # 会话正忙与“会话属于他人”同消息：不泄露归属信息。
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NotConfigured as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # 每租户配额（隔离方案 P0-6）。两次检查都放在 mark_busy 之前，且与它
        # 之间没有 await，因此在单事件循环内是原子的：并发流上限不会因为两次
        # 请求交错而双双通过。检查本身只读注册表现状，失败不会留下 busy 残留。
        cap = settings.quota.max_concurrent_streams
        if cap and registry.busy_count(user.id) >= cap:
            observer.record_governance_event(kind="quota_concurrent_streams")
            raise HTTPException(
                status_code=429,
                detail=f"同时进行的对话过多（上限 {cap}），请等待其中一轮结束",
                headers={"Retry-After": "5"},
            )
        turn_decision = turn_limiter.check(f"turns:{user.id}")
        if not turn_decision.allowed:
            observer.record_governance_event(kind="quota_turn_budget")
            raise HTTPException(
                status_code=429,
                detail=f"本时段对话轮次已达上限，请 {turn_decision.retry_after_s} 秒后重试",
                headers={"Retry-After": str(turn_decision.retry_after_s)},
            )

        # Explicit resume routing (before mark_busy): FSM rows are authoritative.
        # Resumable snapshot → resume in place; no rows + recoverable checkpoint
        # → new FSM hydrate; any non-resumable FSM row → 409 (no checkpoint fallback).
        loop_resume = False
        run_message = request.message
        if request.resume:
            agent_states = SqliteAgentStateStore(memory_store)
            latest = agent_states.latest(session.conversation_id, user.id)
            if latest is not None:
                if not _snapshot_resumable(latest):
                    raise HTTPException(
                        status_code=409,
                        detail="no resumable agent state for this conversation",
                    )
                run_message = request.message or ""
                loop_resume = True
            else:
                checkpoint = memory_store.load_latest_checkpoint(
                    session.conversation_id
                )
                if checkpoint is None or not checkpoint.recoverable:
                    raise HTTPException(
                        status_code=409,
                        detail="no resumable agent state for this conversation",
                    )
                run_message = request.message or ""
                loop_resume = False

        sink = QueueSink()
        session.loop.output = sink
        # 停止信号挂在会话上，使 POST /v1/chat/stop 能从此处之外置位它；
        # 引擎在每个等待点检查它（docs 03.3）。每次请求新建一个，避免上一次
        # 请求的停止状态泄漏到下一次。
        stop_signal = StopSignal()
        session.stop_signal = stop_signal
        session.loop.stop_signal = stop_signal
        registry.mark_busy(session)
        persisted_before = memory_store.message_seq_range(session.conversation_id)[1]
        audit = getattr(session.loop, "audit", None)
        if audit is not None:
            # 审计写失败不能拦住这一轮对话，但也必须可见（隔离方案 P0-8）。
            try:
                audit.session_start(
                    mode=settings.permission.default_mode,
                    provider=type(session.loop.provider).__name__,
                    model=getattr(session.loop.provider, "model", ""),
                )
            except Exception:  # noqa: BLE001 - 审计失败不阻断会话
                get_logger("finharness.server.api").exception("session_start_audit_failed")
        # trace_id 在请求入口生成并绑定到当前上下文；``create_task`` 会复制
        # context，因此引擎任务与所有日志自动携带同一个 id。
        trace_ctx = observer.bind_request(
            session_id=session.session_id,
            conversation_id=session.conversation_id,
        )
        # NullObserver（或旧签名）的 bind_request 返回空串：此时引擎内部
        # 仍会自建 trace id，服务端用 session 派生一个等价稳定的 run_id。
        run_id = getattr(trace_ctx, "trace_id", "") or f"tr_{session.session_id}"

        # 监控平台旁路（docs 03.14.4）：开启 trace_store 时，本轮对话的全部
        # 非文本事件、轮次轨迹与终态都落库。失败只降级（TraceStore 吞错）。
        trace_run_started = trace_store is not None
        if trace_store is not None:
            await asyncio.to_thread(
                trace_store.start_run,
                run_id=run_id,
                source="server",
                user_id=user.id,
                session_id=session.session_id,
                conversation_id=session.conversation_id,
                input=request.message,
            )

            def record_trace_event(event) -> None:
                if event.kind != "text_delta":
                    trace_store.record_event(run_id, event.kind, dict(event.data))

            def record_trace_round(round_trace, *, phase=None, revision=None) -> None:
                trace_store.record_round(run_id, round_trace, phase=phase, revision=revision)

        else:
            record_trace_event = None
            record_trace_round = None
        sink = QueueSink(trace_recorder=record_trace_event, round_recorder=record_trace_round)
        session.loop.output = sink
        task = asyncio.create_task(
            session.loop.run(run_message, resume=loop_resume)
        )

        async def events() -> AsyncIterator[str]:
            """SSE 事件生成器：转发引擎事件并在结束时补齐元数据。

            请求级的耗时/根 Span 由 ``AgentLoop.run`` 持有——它才是服务端、
            eval 与脚本共同的执行边界，在这里再开一个只会重复计数。本层只
            负责绑定 trace id 与传输。

            终止保证：无论引擎是正常结束、抛出异常，还是在没有下发 ``done``
            的情况下退出，本生成器都保证给客户端一个终止事件（必要时先补一个
            ``error``）。否则浏览器只能看到连接被静默关闭，界面永久停在
            "运行中"——那正是"刷新后才看到结果"的成因。两次事件之间的静默
            由心跳帧兜底。
            """
            log = get_logger("finharness.server.api")
            started = time.monotonic()
            frames = 0
            terminal_sent = False
            settled = False
            engine_error: BaseException | None = None

            # 两个 id 都会传递：客户端持久化 conversation_id（持久），
            # 并可能回传 session_id 以在同一窗口内提高效率。
            yield encode_event(
                "session",
                {
                    "session_id": session.session_id,
                    "conversation_id": session.conversation_id,
                },
            )
            log.info(
                "chat_stream_open",
                extra={"conversation_id": session.conversation_id, "session_id": session.session_id},
            )

            async def settle_engine() -> None:
                """等待引擎收尾并捕获其异常；只结算一次，且绝不让异常逃逸。

                引擎的异常必须在这里被"取走"：否则它既会以"异常未被读取"
                的形式留在任务上，也会在结算路径上重复抛出。
                """
                nonlocal settled, engine_error
                if settled:
                    return
                settled = True
                try:
                    await task
                except (asyncio.CancelledError, GeneratorExit):
                    raise
                except BaseException as exc:  # noqa: BLE001 - 收尾失败只降级为日志
                    engine_error = exc
                    log.exception("chat_stream_engine_failed")

            def engine_failure() -> BaseException | None:
                if engine_error is not None:
                    return engine_error
                if task.done() and not task.cancelled():
                    try:
                        return task.exception()
                    except BaseException:  # noqa: BLE001 - 取不到即视为无异常
                        return None
                return None

            async def finish_trace(status: str, reason: str | None = None) -> None:
                """把本次运行的终态与轮次轨迹落库（未启用时为 no-op）。

                覆盖四条收尾路径：done 事件、引擎异常、用户停止/断连、以及
                引擎正常返回但缺少终止事件的兜底。outcome 的 ``trace``（每轮
                thought/actions/observations）只在引擎正常返回时可得；异常路径
                落一个终态行即可——事件流已经把过程留下来了。

                用量账本在 trace **之前**落一行：它是计费口径，与 trace 开关
                解耦——监控关闭时 token/轮次照记不误。落账失败由 UsageStore
                吞掉（旁路纪律），绝不影响收尾。写入在工作线程执行，避免占用
                事件循环做磁盘 I/O。
                """
                outcome = None
                if task.done() and not task.cancelled():
                    try:
                        outcome = task.result()
                    except BaseException:  # noqa: BLE001 - 异常路径没有 outcome
                        outcome = None
                succeeded = None
                if outcome is not None:
                    succeeded = bool(getattr(outcome, "succeeded", False))
                elif status == "done":
                    succeeded = True
                usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_hit_tokens": 0,
                }
                if outcome is not None:
                    u = getattr(outcome, "usage", None)
                    usage = {
                        "input_tokens": getattr(u, "input_tokens", 0) or 0,
                        "output_tokens": getattr(u, "output_tokens", 0) or 0,
                        "cache_hit_tokens": getattr(u, "cache_hit_tokens", 0) or 0,
                    }
                elapsed_ms = round((time.monotonic() - started) * 1000)
                await asyncio.to_thread(
                    usage_store.record_turn,
                    user_id=user.id,
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    cache_hit_tokens=usage["cache_hit_tokens"],
                    duration_ms=elapsed_ms,
                    # 账本只关心计费口径的终态：done/stopped/error。
                    status=status,
                )
                if trace_store is None or not trace_run_started:
                    return
                await asyncio.to_thread(
                    trace_store.finish_run,
                    run_id,
                    status=status,
                    answer=str(getattr(outcome, "answer", "") or ""),
                    reason=reason or (getattr(outcome, "reason", None) if outcome else None),
                    succeeded=succeeded,
                    rounds=getattr(outcome, "rounds", None) if outcome else None,
                    tool_calls=getattr(outcome, "tool_calls", None) if outcome else None,
                    retry_count=getattr(outcome, "retry_count", None) if outcome else None,
                    usage=usage,
                    citations=list(getattr(outcome, "citations", []) or []) if outcome else None,
                    trace_rounds=list(getattr(outcome, "trace", []) or []) if outcome else None,
                )

            async def terminal_frames() -> list[str]:
                """引擎未下发终止事件时的兜底帧，保证客户端一定收敛。"""
                nonlocal terminal_sent
                terminal_sent = True
                failure = engine_failure()
                await finish_trace(
                    "error" if failure is not None else "done",
                    "engine_error" if failure is not None else "missing_terminal_event",
                )
                if failure is None:
                    # 引擎正常返回却没有 ``done``：这是异常路径，但答案通常
                    # 已落盘，因此对用户按成功收尾，只留一条告警供排查。
                    log.warning(
                        "chat_stream_missing_terminal_event",
                        extra={"elapsed_ms": round((time.monotonic() - started) * 1000)},
                    )
                    return [
                        encode_event(
                            "done",
                            {
                                "succeeded": True,
                                "reason": None,
                                "session_id": session.session_id,
                                "conversation_id": session.conversation_id,
                            },
                        )
                    ]
                log.error(
                    "chat_stream_engine_error",
                    extra={
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                        "error_type": type(failure).__name__,
                    },
                )
                message = f"引擎执行失败：{type(failure).__name__}"
                return [
                    encode_event("error", {"reason": "engine_error", "message": message}),
                    encode_event(
                        "done",
                        {
                            "succeeded": False,
                            "reason": "engine_error",
                            "message": message,
                            "session_id": session.session_id,
                            "conversation_id": session.conversation_id,
                        },
                    ),
                ]

            try:
                while True:
                    getter = asyncio.ensure_future(sink.queue.get())
                    try:
                        # 同时等待"下一个事件"与"引擎结束"：把 task 放进等待集，
                        # 引擎不再入队任何事件时循环立即可退，消除了旧实现里
                        # "先检查 done 再阻塞 get"的竞态（该竞态能把生成器
                        # 永远挂在 queue.get 上）。
                        done_set, _ = await asyncio.wait(
                            {getter, task},
                            timeout=HEARTBEAT_S,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    except BaseException:
                        if not getter.done():
                            getter.cancel()
                            with contextlib.suppress(BaseException):
                                await getter
                        raise

                    if getter in done_set:
                        event = getter.result()
                    else:
                        # 心跳窗口内既没有新事件、引擎也还没结束：下发保活帧。
                        getter.cancel()
                        with contextlib.suppress(BaseException):
                            await getter
                        if task.done() and sink.queue.empty():
                            await settle_engine()
                            if not terminal_sent:
                                for frame in await terminal_frames():
                                    yield frame
                            break
                        # 记录心跳，使"引擎静默了多久"在日志里可见——旧实现下
                        # 这段静默完全没有痕迹，无法与"连接已死"区分。
                        log.debug(
                            "chat_stream_heartbeat",
                            extra={
                                "frames": frames,
                                "elapsed_ms": round((time.monotonic() - started) * 1000),
                            },
                        )
                        yield encode_comment()
                        continue

                    data = event.data
                    if event.kind == "done":
                        data = {
                            **data,
                            "session_id": session.session_id,
                            "conversation_id": session.conversation_id,
                        }
                        # 引擎在将最终回答刷入记忆之前发出 ``done``。
                        # 在浏览器可能观察到完成并刷新页面之前，
                        # 先完成该刷写并附上重放记录；但这一步失败不得
                        # 累终止帧，否则答案已落库、界面却停在运行中。
                        await settle_engine()
                        await finish_trace(
                            "stopped" if stop_signal.requested else "done",
                            str(data.get("reason") or "done"),
                        )
                        turn_metadata = sink.turn_metadata()
                        if turn_metadata is not None:
                            try:
                                await asyncio.to_thread(
                                    memory_store.attach_latest_answer_metadata,
                                    session.conversation_id,
                                    metadata={"turn": turn_metadata},
                                    after_seq=persisted_before,
                                )
                            except Exception:  # noqa: BLE001 - 元数据缺失不该中断流
                                log.exception("chat_stream_attach_metadata_failed")
                        terminal_sent = True
                    # request_id 属于确认总线，而非引擎载荷的形状，
                    # 因此在传输边缘添加。
                    yield encode_event(_event_name(event.kind), data)
                    frames += 1
                    # 只记录有信息量的帧：text_delta 一次可达数千条，逐条落盘
                    # 只会淹没真正需要排查的时序（工具开始/结束、终止帧）。
                    if event.kind != "text_delta":
                        log.debug(
                            "chat_stream_frame",
                            extra={
                                "kind": event.kind,
                                "frames": frames,
                                "elapsed_ms": round((time.monotonic() - started) * 1000),
                            },
                        )
                    if event.kind == "done":
                        break

                await settle_engine()
                if audit is not None:
                    snapshot = session.loop.stats.snapshot()
                    try:
                        audit.session_end(
                            total_tokens=snapshot.input_tokens + snapshot.output_tokens,
                            tool_calls=snapshot.tool_calls,
                        )
                    except Exception:  # noqa: BLE001 - 审计失败不阻断收尾
                        get_logger("finharness.server.api").exception(
                            "session_end_audit_failed"
                        )
                log.info(
                    "chat_stream_close",
                    extra={
                        "frames": frames,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                    },
                )
            except (asyncio.CancelledError, GeneratorExit):
                task.cancel()
                # 中断的流不能一直让模型等待下去。
                confirm_bus.cancel_session(session.session_id)
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                # 终态落库：用户停止（stop_signal 已置位）与传输掐断分开记账，
                # 与下方日志的口径一致。await 在此收尾路径上不会丢失写入：
                # 上游取消只投递一次，to_thread 的任务会照常提交。
                await finish_trace(
                    "stopped" if stop_signal.requested else "aborted",
                    "user_stop" if stop_signal.requested else "disconnected",
                )
                # 区分两种断流：用户自己按了停止（前端宽限期到期后硬断兜底）
                # 与传输/代理掐断。两者的运维含义完全不同，混成一条日志会
                # 让"停止功能坏了吗"无从判断。
                log.warning(
                    "chat_stream_stopped"
                    if stop_signal.requested
                    else "chat_stream_disconnected",
                    extra={
                        "frames": frames,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                    },
                )
                raise
            except Exception:
                # 传输层自身的意外失败：至少给客户端一个终止事件，
                # 而不是静默断流。落盘由引擎负责，这里只负责收尾。
                log.exception("chat_stream_transport_failed")
                await finish_trace("error", "transport_failed")
                if not terminal_sent:
                    for frame in await terminal_frames():
                        yield frame
            finally:
                # 停止信号属于本次请求：清掉它，避免下一次请求一开始就处于
                # "已请求停止"的状态。``release`` 也会清，这里覆盖异常路径。
                session.stop_signal = None
                registry.release(session)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                # 反缓冲：每一帧都应尽快到达浏览器，否则"运行中"与"已完成"
                # 之间的帧会被中间代理攒着一起发，用户看到的就是界面卡住。
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return application


def create_production_app(path: str | Path = "settings.json") -> FastAPI:
    """构建生产用应用：从文件加载设置并校验运行前提后创建应用。"""
    settings = Settings.from_file(path, require_api_key=False)
    # API key 延迟解析（先数据库配置，后环境变量），
    # 因此启动时只校验结构与审计可写性，不校验凭据。
    settings.validate_runtime(require_api_key=False)
    return create_app(settings=settings)


def _event_name(kind: str) -> str:
    return {"text_delta": "delta"}.get(kind, kind)


app = create_app()
