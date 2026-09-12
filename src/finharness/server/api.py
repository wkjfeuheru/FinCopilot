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
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.provider.fake import FakeProvider
from finharness.provider.registry import build_provider
from finharness.provider.resolver import NotConfigured, ProviderResolver
from finharness.server.config_api import create_config_router
from finharness.server.sessions import SessionBusyError, SessionRegistry
from finharness.server.sse import encode_event
from finharness.tools.registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are FinHarness, a careful financial research copilot. "
    "Use only provided tools for factual market data."
)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    mode: str = "default"


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

    def make_registry() -> ToolRegistry:
        return ToolRegistry(shared_data)
    tool_registry = make_registry()
    session_citations: dict[str, CitationRegistry] = {}
    fallback_provider = provider
    def loop_factory(session_id: str | None = None) -> AgentLoop:
        if fallback_provider is not None:
            selected = fallback_provider
        else:
            selected = resolver.current()
        citations = CitationRegistry()
        if session_id:
            session_citations[session_id] = citations
        return AgentLoop(
            provider=selected,
            registry=make_registry(),
            settings=settings,
            system=DEFAULT_SYSTEM_PROMPT,
            cite=citations,
            session_id=session_id,
        )

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
    async def tools() -> dict[str, list]:
        return {"tools": tool_registry.names()}

    @application.get("/v1/cache/stats")
    async def cache_stats() -> dict:
        snapshot = data_cache.stats()
        return {
            "entries": snapshot.entries,
            "hits": snapshot.hits,
            "misses": snapshot.misses,
            "hit_ratio": snapshot.hit_ratio,
        }

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
                    yield encode_event(_event_name(event.kind), data)
                    if event.kind == "done":
                        break
                await task
            except (asyncio.CancelledError, GeneratorExit):
                task.cancel()
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
