"""FastAPI 应用与 M0 聊天路由。

除 ``/v1/health``、``/v1/auth/*`` 与 ``/metrics`` 外，所有端点都要求
认证，且所有数据访问都按当前用户隔离（docs 03.13）。
"""

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from finharness.auth.dependency import create_require_user
from finharness.auth.store import CurrentUser, UserStore
from finharness.config.crypto import SecretCipher
from finharness.config.settings import Settings
from finharness.config.store import ConfigStore
from finharness.context.memory.store import MemoryStore
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.adapters.eastmoney_report_adapter import EastmoneyReportAdapter
from finharness.data.adapters.tavily_adapter import TavilyAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.hooks.audit import AuditHook, AuditLogWriter, summarize_args
from finharness.hooks.base import HookChain
from finharness.observability import build_observer, setup_logging
from finharness.permissions.gate import PermissionGate
from finharness.provider.fake import FakeProvider
from finharness.provider.registry import build_provider
from finharness.provider.resolver import NotConfigured, ProviderResolver
from finharness.server.auth_api import create_auth_router
from finharness.server.config_api import create_config_router
from finharness.server.confirm import ConfirmBus
from finharness.server.sessions import SessionBusyError, SessionRegistry
from finharness.server.sse import encode_event
from finharness.tools.registry import ALL_TOOL_CLASSES, ToolRegistry

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
    mode: str = "default"


class RespondRequest(BaseModel):
    request_id: str
    response: str


class QueueSink:
    def __init__(self):
        import asyncio

        self.queue = asyncio.Queue()
        self.started_at = time.monotonic()
        self.first_token_at: float | None = None
        self.done_at: float | None = None
        self.replay_events: list[dict] = []

    async def emit(self, event):
        """将引擎事件入队，并记录计时与重放所需的元数据。"""
        now = time.monotonic()
        if event.kind == "text_delta" and self.first_token_at is None:
            self.first_token_at = now
        if event.kind == "done":
            self.done_at = now
        if event.kind in {
            "tool_status",
            "context_compacted",
            "loop_guard",
            "plan_progress",
            "interactive_request",
            # ``answer`` 与终态的 ``done`` 也属于失败轮次的重放记录：
            # 重新加载的对话仍须展示本次运行已确立的部分发现。
            "answer",
            "done",
        }:
            self.replay_events.append(
                {"event": _event_name(event.kind), "data": dict(event.data)}
            )
        await self.queue.put(event)

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
) -> FastAPI:
    """构建 FastAPI 应用，装配缓存、适配器、会话注册表与全部路由。"""
    application = FastAPI(title="FinHarness")
    settings = settings or Settings.from_file()

    # 日志在任何组件之前配置，使启动期的日志本身就带上下文字段。
    setup_logging(
        level=settings.observability.logging.level,
        json_format=settings.observability.logging.json_format,
        path=settings.observability.logging.path,
    )
    # 观测门面：未启用或未安装可选依赖时自动降级为 no-op，绝不影响主流程。
    observer = build_observer(settings)

    # -- 认证（docs 03.13）-----------------------------------------------------
    user_store = UserStore(settings.paths.auth_db)
    application.state.user_store = user_store
    require_user = create_require_user(user_store)

    # 对话记忆：一个存储服务所有对话；对话记录、
    # 引用、结论与摘要分段均以 conversation id 为键。
    # （提前构造：首个注册用户的存量认领需要它。）
    memory_store = MemoryStore(settings.paths.memory_db)
    application.state.memory_store = memory_store

    def store_factory() -> ConfigStore:
        nonlocal config_store
        if config_store is None:
            config_store = ConfigStore(
                settings.data.cache_dir / "config.db",
                cipher=SecretCipher(settings.data.cache_dir / "secret.key"),
            )
        return config_store

    def _claim_legacy(user_id: str) -> None:
        """首个注册用户认领单用户时代的存量数据（对话、笔记、供应商配置）。"""
        memory_store.claim_user(user_id)
        store_factory().claim_user(user_id)

    application.include_router(
        create_auth_router(
            store=user_store,
            ttl_s=settings.auth.token_ttl_s,
            secure_cookie=settings.auth.secure_cookie,
            allow_register=settings.auth.allow_register,
            claim_legacy=_claim_legacy,
        )
    )

    if resolver is None:
        resolver = ProviderResolver(store_factory=store_factory, settings=settings)

    # 一个缓存服务整个进程；每个会话有自己的引用注册表，
    # 使来源信息限定在该对话内。调用方提供的 DataAccess 按原样使用
    # （测试传入隔离的替身），因此这里绝不改动它。
    data_cache = LocalCache(settings.data.cache_dir)
    adapters = [
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
        ),
    ]

    shared_data = data_access or DataAccess(
        adapters,
        cache=data_cache,
        settings=settings,
    )

    # 每个用户的工具级 DataAccess：与 shared_data 共享适配器与缓存，
    # 只是 output/ 换成了 output/<user>/，因此用户之间互不可见。
    # 测试注入的 DataAccess 替身按原样使用，不引入用户目录。
    user_data_access: dict[str, DataAccess] = {}

    def data_for(user_id: str) -> DataAccess:
        if data_access is not None:
            return shared_data
        scoped = user_data_access.get(user_id)
        if scoped is None:
            user_settings = settings.model_copy(
                update={
                    "paths": settings.paths.model_copy(
                        update={"output_dir": settings.paths.output_dir / user_id}
                    )
                }
            )
            scoped = DataAccess(adapters, cache=data_cache, settings=user_settings)
            user_data_access[user_id] = scoped
        return scoped

    # 治理：一个确认总线桥接所有会话的交互式请求。
    confirm_bus = ConfirmBus(ttl_s=settings.server.confirm_ttl_s)
    # 暴露出来，使测试与运维人员可以检查待处理的交互式请求。
    application.state.confirm_bus = confirm_bus
    session_citations: dict[str, CitationRegistry] = {}
    session_contexts: dict[str, ResearchContext] = {}
    # 引用也跟随对话，因此产生它们的执行会话过期后仍可寻址。
    conversation_citations: dict[str, CitationRegistry] = {}
    fallback_provider = provider

    memory_store.prune(
        max_conversations=settings.context.retention_conversations,
        max_age_days=settings.context.retention_days,
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
        # 用两个 id 索引：引用与上下文跟随对话，
        # 而 session id 只是本次执行窗口的句柄。
        if session_id:
            session_citations[session_id] = citations
            session_contexts[session_id] = ctx
        if conversation_id:
            conversation_citations[conversation_id] = citations

        # 工具 schema 按会话生成，因为惰性激活是会话作用域的；
        # ctx 与 registry 相互接线。产物经由 data_for(user_id) 写入
        # output/<user>/，因此用户之间互不可见。
        registry_for_session = ToolRegistry(
            data_for(user_id), ctx=ctx, settings=settings
        )
        audit = AuditHook(
            AuditLogWriter(settings.audit.log_path),
            session_id=session_id or "local",
            user_id=user_id,
        )
        gate = PermissionGate(
            settings=settings,
            confirm=lambda name, args: _ask(
                "confirm",
                f"工具 {name} 将执行，入参：{summarize_args(args)}",
                ["y", "n"],
            ),
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
        )

        async def _ask(kind: str, prompt: str, options: list[str]):
            """经由 SSE 宣告请求，然后 await 客户端的应答。"""
            _, answer = await confirm_bus.request(
                session_id=session_id or "local",
                kind=kind,
                prompt=prompt,
                options=options,
                user_id=user_id,
                announce=lambda payload: loop._emit("interactive_request", payload),
            )
            return answer

        # 构造之后注入，使回调可以通过此 loop 发送事件。
        loop.interactive = _ask
        loop.audit = audit
        return loop

    registry = SessionRegistry(loop_factory, ttl_s=settings.server.session_ttl_s)
    application.state.session_registry = registry
    application.include_router(
        create_config_router(
            store_factory=store_factory,
            resolver=resolver,
            settings=settings,
            probe_client_factory=probe_client_factory,
            require_user=require_user,
        )
    )
    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if (frontend_dist / "assets").is_dir():
        application.mount(
            "/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets"
        )

    @application.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    metrics_recorder = getattr(observer, "metrics", None)
    if metrics_recorder is not None:

        @application.get("/metrics")
        async def metrics_endpoint() -> Response:
            """Prometheus 抓取端点；仅在 metrics 启用时注册。"""
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

    @application.get("/v1/cache/stats")
    async def cache_stats(user: CurrentUser = Depends(require_user)) -> dict:
        """返回本地数据缓存的命中统计。"""
        snapshot = data_cache.stats()
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
        user: CurrentUser = Depends(require_user),
    ) -> dict:
        """累积记忆的只读视图。

        取代了设计曾提出的 MEMORY.md 文件视图：Web 层正是人类阅读
        它的地方，而且它保持结构化，而不是每轮都重写到一个文件。
        """
        return {
            "notes": memory_store.get_notes(user_id=user.id),
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

    @application.get("/v1/artifacts")
    async def download_artifact(
        path: str, user: CurrentUser = Depends(require_user)
    ):
        """提供产出的文件，范围限于该用户自己的产物目录与缓存数据。

        在解析之后强制校验包含关系，与 read_file 一致；没有它，
        该 endpoint 就会变成任意文件读取。缓存侧只开放 ``parquet/``
        子树——打开整个 data_cache/ 会连带暴露 users.db（口令哈希）
        与 memory.db（全部用户的对话）。
        """
        allowed_roots = [
            (Path(settings.paths.output_dir) / user.id).resolve(),
            (Path(settings.data.cache_dir) / "parquet").resolve(),
        ]
        try:
            target = Path(path).resolve()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="非法路径") from exc
        if not any(
            target == root or target.is_relative_to(root) for root in allowed_roots
        ):
            raise HTTPException(
                status_code=403,
                detail="路径超出允许范围（仅限 output/ 与 data_cache/）",
            )
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
            registry = conversation_citations.get(conversation_id)
            if registry is not None:
                records = registry.all()
            else:
                # 对话比服务它的进程更长寿，因此回退到持久化的引用：
                # 恢复的对话即使在服务端重启后仍能显示其来源。
                records = memory_store.load_citations(conversation_id)
        elif session_id:
            session = registry.sessions.get(session_id)
            if session is None or session.user_id != user.id:
                raise HTTPException(status_code=404, detail="会话或对话不存在")
            records = session_citations[session_id].all()
        else:
            # 无过滤参数：只汇总该用户各会话的引用。
            records = [
                item
                for sid, s in registry.sessions.items()
                if s.user_id == user.id
                for item in session_citations[sid].all()
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
        return {
            "conversation_id": conversation_id,
            "messages": rendered[-max(limit, 1) :],
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
        memory_store.delete_conversation(conversation_id)
        conversation_citations.pop(conversation_id, None)
        for session_id, session in list(registry.sessions.items()):
            if session.conversation_id == conversation_id:
                registry.sessions.pop(session_id, None)
                session_citations.pop(session_id, None)
                session_contexts.pop(session_id, None)
        return {"ok": True, "conversation_id": conversation_id}

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
        sink = QueueSink()
        session.loop.output = sink
        session.busy = True
        persisted_before = memory_store.message_seq_range(session.conversation_id)[1]
        audit = getattr(session.loop, "audit", None)
        if audit is not None:
            audit.session_start(
                mode=settings.permission.default_mode,
                provider=type(session.loop.provider).__name__,
                model=getattr(session.loop.provider, "model", ""),
            )
        # trace_id 在请求入口生成并绑定到当前上下文；``create_task`` 会复制
        # context，因此引擎任务与所有日志自动携带同一个 id。
        observer.bind_request(
            session_id=session.session_id,
            conversation_id=session.conversation_id,
            mode=request.mode,
        )
        task = asyncio.create_task(session.loop.run(request.message))

        async def events() -> AsyncIterator[str]:
            """SSE 事件生成器：转发引擎事件并在结束时补齐元数据。

            请求级的耗时/根 Span 由 ``AgentLoop.run`` 持有——它才是服务端、
            eval 与脚本共同的执行边界，在这里再开一个只会重复计数。本层只
            负责绑定 trace id 与传输。
            """
            # 两个 id 都会传递：客户端持久化 conversation_id（持久），
            # 并可能回传 session_id 以在同一窗口内提高效率。
            yield encode_event(
                "session",
                {
                    "session_id": session.session_id,
                    "conversation_id": session.conversation_id,
                },
            )
            try:
                while True:
                    if task.done() and sink.queue.empty():
                        break
                    event = await sink.queue.get()
                    data = event.data
                    if event.kind == "done":
                        data = {
                            **data,
                            "session_id": session.session_id,
                            "conversation_id": session.conversation_id,
                        }
                        # 引擎在将最终回答刷入记忆之前发出 ``done``。
                        # 在浏览器可能观察到完成并刷新页面之前，
                        # 先完成该刷写并附上重放记录。
                        await task
                        turn_metadata = sink.turn_metadata()
                        if turn_metadata is not None:
                            memory_store.attach_latest_answer_metadata(
                                session.conversation_id,
                                metadata={"turn": turn_metadata},
                                after_seq=persisted_before,
                            )
                    # request_id 属于确认总线，而非引擎载荷的形状，
                    # 因此在传输边缘添加。
                    yield encode_event(_event_name(event.kind), data)
                    if event.kind == "done":
                        break
                await task
                if audit is not None:
                    snapshot = session.loop.stats.snapshot()
                    audit.session_end(
                        total_tokens=snapshot.input_tokens + snapshot.output_tokens,
                        tool_calls=snapshot.tool_calls,
                    )
            except (asyncio.CancelledError, GeneratorExit):
                task.cancel()
                # 中断的流不能一直让模型等待下去。
                confirm_bus.cancel_session(session.session_id)
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise
            finally:
                registry.release(session)

        return StreamingResponse(events(), media_type="text/event-stream")

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
