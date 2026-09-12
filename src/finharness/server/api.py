"""FastAPI application and M0 chat routes."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from finharness.config.crypto import SecretCipher
from finharness.config.settings import Settings
from finharness.config.store import ConfigStore
from finharness.engine.loop import AgentLoop
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
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

    tool_registry = ToolRegistry(data_access or DataAccess([AkShareAdapter()]))
    fallback_provider = provider

    def loop_factory() -> AgentLoop:
        if fallback_provider is not None:
            selected = fallback_provider
        else:
            selected = resolver.current()
        return AgentLoop(
            provider=selected,
            registry=tool_registry,
            settings=settings,
            system=DEFAULT_SYSTEM_PROMPT,
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
