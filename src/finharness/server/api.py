"""FastAPI application and M0 chat routes."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from finharness.config.settings import Settings
from finharness.engine.loop import AgentLoop
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.provider.fake import FakeProvider
from finharness.provider.registry import build_provider
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


def create_app(provider=None, data_access=None, settings: Settings | None = None) -> FastAPI:
    application = FastAPI(title="FinHarness")
    settings = settings or Settings.from_file()
    selected_provider = provider or FakeProvider([], error=RuntimeError(
        "No provider configured. Start the production app factory with DEEPSEEK_API_KEY, or inject FakeProvider explicitly for tests."
    ))
    tool_registry = ToolRegistry(data_access or DataAccess([AkShareAdapter()]))
    registry = SessionRegistry(
        lambda: AgentLoop(
            provider=selected_provider,
            registry=tool_registry,
            settings=settings,
            system=DEFAULT_SYSTEM_PROMPT,
        ),
        ttl_s=settings.server.session_ttl_s,
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
        try:
            session = await registry.ensure(request.session_id)
        except SessionBusyError as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
    settings = Settings.from_file(path)
    settings.validate_runtime()
    return create_app(build_provider(settings=settings), settings=settings)


def _event_name(kind: str) -> str:
    return {"text_delta": "delta"}.get(kind, kind)


app = create_app()
