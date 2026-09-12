"""FastAPI application and M0 chat routes."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from finharness.config.crypto import SecretCipher
from finharness.config.settings import Settings
from finharness.config.store import ConfigStore
from finharness.context.memory.store import MemoryStore
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.hooks.audit import AuditHook, AuditLogWriter, summarize_args
from finharness.hooks.base import HookChain
from finharness.permissions.gate import PermissionGate
from finharness.provider.fake import FakeProvider
from finharness.provider.registry import build_provider
from finharness.provider.resolver import NotConfigured, ProviderResolver
from finharness.server.config_api import create_config_router
from finharness.server.confirm import ConfirmBus
from finharness.server.sessions import SessionBusyError, SessionRegistry
from finharness.server.sse import encode_event
from finharness.tools.registry import ALL_TOOL_CLASSES, ToolRegistry

# One definition, loaded from prompts/system.md so tests exercise the same text
# the product ships (docs: it governs planning, citation and convergence).
DEFAULT_SYSTEM_PROMPT = system_prompt()


class ChatRequest(BaseModel):
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

    async def emit(self, event):
        await self.queue.put(event)


def create_app(
    provider=None,
    data_access=None,
    settings: Settings | None = None,
    *,
    config_store: ConfigStore | None = None,
    resolver: ProviderResolver | None = None,
    probe_client_factory=None,
) -> FastAPI:
    application = FastAPI(title="FinHarness")
    settings = settings or Settings.from_file()

    def store_factory() -> ConfigStore:
        nonlocal config_store
        if config_store is None:
            config_store = ConfigStore(
                settings.data.cache_dir / "config.db",
                cipher=SecretCipher(settings.data.cache_dir / "secret.key"),
            )
        return config_store

    if resolver is None:
        resolver = ProviderResolver(store_factory=store_factory, settings=settings)

    # One cache serves the whole process; each session gets its own citation
    # registry so provenance stays scoped to that conversation. A caller-supplied
    # DataAccess is used as-is (tests pass hermetic doubles), so it is never
    # mutated here.
    data_cache = LocalCache(settings.data.cache_dir)
    shared_data = data_access or DataAccess(
        [AkShareAdapter(throttle_seconds=settings.data.throttle_seconds)],
        cache=data_cache,
        settings=settings,
    )

    # Governance: one confirm bus bridges every session's interactive requests.
    confirm_bus = ConfirmBus(ttl_s=settings.server.confirm_ttl_s)
    # Exposed so tests and operators can inspect pending interactive requests.
    application.state.confirm_bus = confirm_bus
    session_citations: dict[str, CitationRegistry] = {}
    session_contexts: dict[str, ResearchContext] = {}
    fallback_provider = provider

    # Conversation memory: one store serves every conversation; the transcript,
    # citations, conclusions and summary segments are keyed by conversation id.
    memory_store = MemoryStore(settings.paths.memory_db)
    application.state.memory_store = memory_store
    memory_store.prune(
        max_conversations=settings.context.retention_conversations,
        max_age_days=settings.context.retention_days,
    )

    def loop_factory(session_id: str | None = None) -> AgentLoop:
        if fallback_provider is not None:
            selected = fallback_provider
        else:
            selected = resolver.current()
        citations = CitationRegistry()
        ctx = ResearchContext(cite=citations, settings=settings)
        if session_id:
            session_citations[session_id] = citations
            session_contexts[session_id] = ctx

        # Tool schemas are generated per session because lazy activation is
        # session-scoped; ctx and registry are wired to each other.
        registry_for_session = ToolRegistry(shared_data, ctx=ctx, settings=settings)
        audit = AuditHook(
            AuditLogWriter(settings.audit.log_path), session_id=session_id or "local"
        )
        gate = PermissionGate(
            settings=settings,
            confirm=lambda name, args: _ask(
                "confirm", f"工具 {name} 将执行，入参：{summarize_args(args)}", ["y", "n"]
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
            conversation_id=session_id,
            store=memory_store,
        )

        async def _ask(kind: str, prompt: str, options: list[str]):
            """Announce the request over SSE, then await the client's answer."""
            _, answer = await confirm_bus.request(
                session_id=session_id or "local",
                kind=kind,
                prompt=prompt,
                options=options,
                announce=lambda payload: loop._emit("interactive_request", payload),
            )
            return answer

        # Injected after construction so the callback can emit through this loop.
        loop.interactive = _ask
        loop.audit = audit
        return loop

    registry = SessionRegistry(loop_factory, ttl_s=settings.server.session_ttl_s)
    application.include_router(
        create_config_router(
            store_factory=store_factory,
            resolver=resolver,
            settings=settings,
            probe_client_factory=probe_client_factory,
        )
    )
    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if (frontend_dist / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @application.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/tools")
    async def tools(session_id: str | None = None) -> dict:
        """List the catalogue; with a session id, report its activation state."""
        if session_id and session_id in registry.sessions:
            reg = registry.sessions[session_id].loop.registry
            return {
                "tools": reg.names(),
                "resident": reg.resident_names(),
                "lazy": reg.lazy_names(),
                "active": [name for name in reg.names() if reg.is_active(name)],
            }
        return {"tools": [cls.name for cls in ALL_TOOL_CLASSES]}

    @application.post("/v1/chat/respond")
    async def chat_respond(body: RespondRequest) -> dict:
        """Answer a pending interactive request (write confirmation / ask_user)."""
        ok = confirm_bus.respond(request_id=body.request_id, value=body.response)
        if not ok:
            raise HTTPException(status_code=404, detail="请求不存在或已超时")
        return {"ok": True}

    @application.get("/v1/cache/stats")
    async def cache_stats() -> dict:
        snapshot = data_cache.stats()
        return {
            "entries": snapshot.entries,
            "hits": snapshot.hits,
            "misses": snapshot.misses,
            "hit_ratio": snapshot.hit_ratio,
        }

    @application.get("/v1/memory")
    async def memory_view(conversation_id: str | None = None, limit: int = 20) -> dict:
        """Read-only view of accumulated memory.

        Replaces the MEMORY.md file view the design once proposed: the web layer
        is where a human reads this, and it stays structured rather than being
        rewritten to a file on every turn.
        """
        return {
            "notes": memory_store.get_notes(),
            "conversations": [
                {
                    "conversation_id": record.conversation_id,
                    "title": record.title,
                    "created_at": record.created_at,
                    "last_active_at": record.last_active_at,
                }
                for record in memory_store.list_conversations(limit=limit)
            ],
            "conclusions": (
                [
                    {
                        "subject": item.subject,
                        "text": item.text,
                        "cids": list(item.cids),
                        "ts": item.ts,
                    }
                    for item in memory_store.load_conclusions(conversation_id, limit=limit)
                ]
                if conversation_id
                else []
            ),
        }

    @application.get("/v1/artifacts")
    async def download_artifact(path: str):
        """Serve a produced file, restricted to the artefact directories.

        Containment is enforced after resolution, mirroring read_file; without
        it this endpoint would be an arbitrary file read.
        """
        allowed_roots = [
            Path(settings.paths.output_dir).resolve(),
            Path(settings.data.cache_dir).resolve(),
        ]
        try:
            target = Path(path).resolve()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="非法路径") from exc
        if not any(target == root or target.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="路径超出允许范围（仅限 output/ 与 data_cache/）")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(target, filename=target.name)

    @application.get("/v1/citations")
    async def citations(session_id: str | None = None) -> dict:
        """Citations for one session; falls back to every live session."""
        if session_id:
            registries = [session_citations[session_id]] if session_id in session_citations else []
            if not registries:
                raise HTTPException(status_code=404, detail="会话不存在或已回收")
        else:
            registries = list(session_citations.values())
        items: list[dict] = []
        for registry in registries:
            items.extend(
                {
                    "cid": item.cid,
                    "tool": item.tool,
                    "endpoint": item.endpoint,
                    "symbol": item.symbol,
                    "rows": item.rows,
                    "cols": item.cols,
                    "from_cache": item.from_cache,
                    "ts": item.ts,
                    "fingerprint": item.fingerprint,
                }
                for item in registry.all()
            )
        return {"citations": items, "count": len(items)}

    @application.get("/")
    async def root() -> HTMLResponse:
        index = frontend_dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse("<html><body><div id='root'>FinHarness</div></body></html>")

    @application.post("/v1/report")
    async def report_stream(request: ChatRequest):
        """Turn the conversation into a report.

        This endpoint adds no fixed pipeline of its own: it submits an ordinary
        request in the session, so the model loads the report-template skill and
        calls write_report exactly as it would if the user had typed it. The
        resulting files then arrive as tool_status attachments.
        """
        prompt = (
            request.message.strip()
            or "请基于本次会话的研究内容生成一份研报，先加载 report-template 技能再成稿。"
        )
        return await chat_stream(ChatRequest(session_id=request.session_id, message=prompt))

    @application.post("/v1/chat/stream")
    async def chat_stream(request: ChatRequest):
        from fastapi import HTTPException

        try:
            session = await registry.ensure(request.session_id)
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NotConfigured as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        sink = QueueSink()
        session.loop.output = sink
        session.busy = True
        audit = getattr(session.loop, "audit", None)
        if audit is not None:
            audit.session_start(
                mode=settings.permission.default_mode,
                provider=type(session.loop.provider).__name__,
                model=getattr(session.loop.provider, "model", ""),
            )
        task = asyncio.create_task(session.loop.run(request.message))

        async def events() -> AsyncIterator[str]:
            yield encode_event("session", {"session_id": session.session_id})
            try:
                while True:
                    if task.done() and sink.queue.empty():
                        break
                    event = await sink.queue.get()
                    data = event.data
                    if event.kind == "done":
                        data = {**data, "session_id": session.session_id}
                    # The request_id belongs to the confirm bus, not the engine
                    # payload shape, so it is added at the transport edge.
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
                # A dropped stream must not leave the model waiting forever.
                confirm_bus.cancel_session(session.session_id)
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise
            finally:
                registry.release(session)

        return StreamingResponse(events(), media_type="text/event-stream")

    return application


def create_production_app(path: str | Path = "settings.json") -> FastAPI:
    settings = Settings.from_file(path, require_api_key=False)
    # The API key is resolved lazily (database config first, then environment),
    # so startup validates structure and audit writability but not credentials.
    settings.validate_runtime(require_api_key=False)
    return create_app(settings=settings)


def _event_name(kind: str) -> str:
    return {"text_delta": "delta"}.get(kind, kind)


app = create_app()
