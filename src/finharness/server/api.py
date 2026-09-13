"""FastAPI application and M0 chat routes."""

import asyncio
import contextlib
import os
import time
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
from finharness.data.adapters.tavily_adapter import TavilyAdapter
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
    # conversation_id is the memory scope and the client's durable handle; it can
    # be supplied to resume a conversation whose session has expired.
    conversation_id: str | None = None
    # session_id is the live execution window; optional and normally omitted,
    # since the server resolves it from the conversation.
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
        now = time.monotonic()
        if event.kind == "text_delta" and self.first_token_at is None:
            self.first_token_at = now
        if event.kind == "done":
            self.done_at = now
        if event.kind in {
            "tool_status",
            "context_compacted",
            "loop_guard",
            "interactive_request",
            "done",
        }:
            self.replay_events.append(
                {"event": _event_name(event.kind), "data": dict(event.data)}
            )
        await self.queue.put(event)

    def turn_metadata(self) -> dict | None:
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
        [
            AkShareAdapter(throttle_seconds=settings.data.throttle_seconds),
            # Web access rides the same adapter chain and cache; an absent key is
            # fine at startup — the tools report "not configured" when called.
            # An inline key in settings.json wins over the environment variable,
            # so a locally configured key works without exporting anything.
            TavilyAdapter(
                api_key=settings.search.api_key
                or (os.getenv(settings.search.env_key) if settings.search.env_key else None),
                base_url=settings.search.base_url,
                timeout_s=settings.search.timeout_s,
            ),
        ],
        cache=data_cache,
        settings=settings,
    )

    # Governance: one confirm bus bridges every session's interactive requests.
    confirm_bus = ConfirmBus(ttl_s=settings.server.confirm_ttl_s)
    # Exposed so tests and operators can inspect pending interactive requests.
    application.state.confirm_bus = confirm_bus
    session_citations: dict[str, CitationRegistry] = {}
    session_contexts: dict[str, ResearchContext] = {}
    # Citations also follow the conversation, so they remain addressable after
    # the execution session that produced them has expired.
    conversation_citations: dict[str, CitationRegistry] = {}
    fallback_provider = provider

    # Conversation memory: one store serves every conversation; the transcript,
    # citations, conclusions and summary segments are keyed by conversation id.
    memory_store = MemoryStore(settings.paths.memory_db)
    application.state.memory_store = memory_store
    memory_store.prune(
        max_conversations=settings.context.retention_conversations,
        max_age_days=settings.context.retention_days,
    )

    def loop_factory(
        session_id: str | None = None, conversation_id: str | None = None
    ) -> AgentLoop:
        if fallback_provider is not None:
            selected = fallback_provider
        else:
            selected = resolver.current()
        citations = CitationRegistry()
        ctx = ResearchContext(cite=citations, settings=settings)
        # Index by both ids: citations and context follow the conversation, and
        # the session id is just this execution window's handle.
        if session_id:
            session_citations[session_id] = citations
            session_contexts[session_id] = ctx
        if conversation_id:
            conversation_citations[conversation_id] = citations

        # Tool schemas are generated per session because lazy activation is
        # session-scoped; ctx and registry are wired to each other.
        registry_for_session = ToolRegistry(shared_data, ctx=ctx, settings=settings)
        audit = AuditHook(
            AuditLogWriter(settings.audit.log_path), session_id=session_id or "local"
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
    application.state.session_registry = registry
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
        application.mount(
            "/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets"
        )

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
                    for item in memory_store.load_conclusions(
                        conversation_id, limit=limit
                    )
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
        session_id: str | None = None, conversation_id: str | None = None
    ) -> dict:
        """Citations for a conversation (preferred) or a live session."""
        if conversation_id:
            registry = conversation_citations.get(conversation_id)
            if registry is not None:
                records = registry.all()
            else:
                # A conversation outlives the process that served it, so fall back
                # to the persisted citations: a resumed conversation keeps showing
                # its sources even after a server restart.
                if memory_store.get_conversation(conversation_id) is None:
                    raise HTTPException(status_code=404, detail="会话或对话不存在")
                records = memory_store.load_citations(conversation_id)
        elif session_id:
            registry = session_citations.get(session_id)
            if registry is None:
                raise HTTPException(status_code=404, detail="会话或对话不存在")
            records = registry.all()
        else:
            records = [
                item
                for registry in session_citations.values()
                for item in registry.all()
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
    async def conversations(limit: int = 50) -> dict:
        """List stored conversations, newest activity first, for a picker."""
        return {
            "conversations": [
                {
                    "conversation_id": record.conversation_id,
                    "title": record.title,
                    "created_at": record.created_at,
                    "last_active_at": record.last_active_at,
                }
                for record in memory_store.list_conversations(limit=limit)
            ]
        }

    @application.get("/v1/conversations/{conversation_id}/messages")
    async def conversation_messages(conversation_id: str, limit: int = 200) -> dict:
        """Replay a conversation's transcript so a client can restore its view.

        Only user turns and final answers are returned: the intermediate tool
        frames are working state, not something a reader should scroll through.
        """
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
    async def delete_conversation(conversation_id: str) -> dict:
        """Delete a conversation and everything scoped to it.

        Removes the transcript, summaries, citations, conclusions and symbol
        pool. User preferences are global and deliberately unaffected.

        Refused while the conversation is mid-request: deleting the store out
        from under a running loop would leave it persisting into nothing.
        """
        if memory_store.get_conversation(conversation_id) is None:
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
        return await chat_stream(
            ChatRequest(session_id=request.session_id, message=prompt)
        )

    @application.post("/v1/chat/stream")
    async def chat_stream(request: ChatRequest):
        from fastapi import HTTPException

        try:
            session = await registry.ensure(
                request.session_id, conversation_id=request.conversation_id
            )
        except SessionBusyError as exc:
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
        task = asyncio.create_task(session.loop.run(request.message))

        async def events() -> AsyncIterator[str]:
            # Both ids travel: the client persists conversation_id (durable) and
            # may echo session_id back for efficiency within one window.
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
                        # The engine emits ``done`` just before it flushes the
                        # final answer to memory. Finish that flush and attach
                        # the replay record before the browser can observe
                        # completion and refresh the page.
                        await task
                        turn_metadata = sink.turn_metadata()
                        if turn_metadata is not None:
                            memory_store.attach_latest_answer_metadata(
                                session.conversation_id,
                                metadata={"turn": turn_metadata},
                                after_seq=persisted_before,
                            )
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
