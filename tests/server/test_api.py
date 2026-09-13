import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from finharness.config.settings import Settings, SettingsError
from finharness.data.access import DataAccess, RawData
from finharness.data.adapters.base import DataAdapter
from finharness.provider.base import Provider
from finharness.provider.fake import FakeProvider
from finharness.server.api import (
    DEFAULT_SYSTEM_PROMPT,
    app,
    create_app,
    create_production_app,
)
from finharness.types import ModelUsage, StreamChunk, StreamEvent, ToolUse


class ScriptedProvider(Provider):
    """Offline provider double replaying canned rounds per request."""

    def __init__(self, rounds: list[list[StreamChunk] | Exception]):
        self.rounds = list(rounds)
        self.requests: list[list] = []
        self.systems: list[str] = []

    async def stream(
        self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage
    ):
        self.requests.append(list(messages))
        self.systems.append(system)
        script = self.rounds.pop(0) if self.rounds else []
        if isinstance(script, Exception):
            raise script
        for chunk in script:
            yield chunk


class QuoteAdapter(DataAdapter):
    name = "offline"

    def __init__(self):
        self.quoted: list[str] = []

    def fetch_quote(self, symbol):
        import pandas as pd

        from finharness.data.adapters.base import FetchResult

        self.quoted.append(symbol)
        return FetchResult(
            df=pd.DataFrame([{"symbol": symbol, "close": 100.0}]),
            interface="offline_quote",
        )


class BlockingQuoteData:
    """Quote source that never returns, so cancellation can be observed."""

    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def quote(self, symbol: str) -> RawData:
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return RawData(kind="df")


def message_end(
    *tool_uses: ToolUse, input_tokens: int = 1, output_tokens: int = 1
) -> StreamChunk:
    return StreamChunk(
        StreamEvent.MESSAGE_END,
        ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_uses=list(tool_uses),
        ),
    )


def text_round(*parts: str) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in parts] + [
        message_end()
    ]


def tool_round(*tool_uses: ToolUse, draft: tuple[str, ...] = ()) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in draft] + [
        message_end(*tool_uses)
    ]


def parse_events(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        name = None
        payload = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
        assert name is not None, frame
        events.append((name, payload))
    return events


def session_id_from(text: str) -> str:
    return text.split('"session_id": "', 1)[1].split('"', 1)[0]


class AsgiStream:
    """Drives one ASGI request as a cancellable task and records SSE frames."""

    def __init__(self, asgi_app, payload: dict):
        self.asgi_app = asgi_app
        self.payload = payload
        self.frames: list[str] = []
        self.task: asyncio.Task | None = None
        self._body_sent = False

    async def _receive(self) -> dict:
        if self._body_sent:
            await asyncio.sleep(30)
            return {"type": "http.disconnect"}
        self._body_sent = True
        return {
            "type": "http.request",
            "body": json.dumps(self.payload).encode(),
            "more_body": False,
        }

    async def _send(self, message: dict) -> None:
        if message["type"] == "http.response.body":
            chunk = message.get("body", b"").decode()
            if chunk:
                self.frames.append(chunk)

    def start(self) -> asyncio.Task:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/chat/stream",
            "raw_path": b"/v1/chat/stream",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
            ],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }
        self.task = asyncio.create_task(self.asgi_app(scope, self._receive, self._send))
        return self.task

    def text(self) -> str:
        return "".join(self.frames)

    async def wait_for(self, needle: str, *, timeout: float = 5.0) -> None:
        async def poll() -> None:
            while needle not in self.text():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(poll(), timeout)


async def wait_until(predicate, *, timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def make_client(provider=None, data_access=None, tmp_path=None) -> TestClient:
    """Build a client whose memory store and cache live in a temp directory.

    Without an explicit settings object the app would use the repository's
    data_cache/memory.db, so every chat test would append conversations to the
    developer's real store.
    """
    if tmp_path is None:
        import tempfile
        from pathlib import Path

        tmp_path = Path(tempfile.mkdtemp(prefix="finharness_test_"))
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "cache" / "memory.db",
        },
    )
    return TestClient(create_app(provider, data_access=data_access, settings=settings))


def test_health_endpoint_returns_service_status() -> None:
    response = TestClient(app).get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_stream_returns_session_and_answer_events() -> None:
    client = make_client(FakeProvider(["hello", " world"]))

    response = client.post("/v1/chat/stream", json={"message": "question"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: session" in response.text
    assert 'event: delta\ndata: {"text": "hello"}' in response.text
    assert 'event: answer\ndata: {"text": "hello world"}' in response.text
    assert "event: done" in response.text


def test_done_event_carries_the_session_id_and_usage() -> None:
    client = make_client(FakeProvider(["hello"]))

    response = client.post("/v1/chat/stream", json={"message": "question"})

    events = parse_events(response.text)
    names = [name for name, _ in events]
    done = events[-1]
    assert names.count("done") == 1
    assert done[0] == "done"
    assert done[1]["session_id"] == session_id_from(response.text)
    assert done[1]["succeeded"] is True
    assert done[1]["usage"]["input_tokens"] == 1
    assert done[1]["usage"]["output_tokens"] == 1
    # Cache split is always reported so a caller can compute a hit rate.
    assert "cache_hit_tokens" in done[1]["usage"]
    assert "cache_miss_tokens" in done[1]["usage"]
    assert done[1]["tool_calls"] == 0


def test_chat_stream_reports_provider_error_and_done_as_sse_events() -> None:
    client = make_client(FakeProvider([], error=RuntimeError("provider down")))

    response = client.post("/v1/chat/stream", json={"message": "question"})

    events = parse_events(response.text)
    names = [name for name, _ in events]
    assert response.status_code == 200
    assert names.count("error") == 1
    assert names.count("done") == 1
    error = next(payload for name, payload in events if name == "error")
    assert "provider down" in error["message"]
    assert error["reason"] == "provider_error"
    done = events[-1][1]
    assert done["succeeded"] is False
    assert done["reason"] == "provider_error"
    assert done["session_id"] == session_id_from(response.text)


def test_tools_endpoint_lists_m1_financial_tools() -> None:
    client = make_client(FakeProvider(["ok"]))

    response = client.get("/v1/tools")

    assert response.json() == {
        "tools": [
            "get_quote",
            "get_kline",
            "get_indicators",
            "get_financials",
            "get_valuation",
            "get_peers",
            "get_market_news",
            "get_announcements",
            "calc_metrics",
            "calc_valuation",
            "make_chart",
            "write_report",
            "read_file",
            "write_file",
            "web_search",
            "fetch_url",
            "research_plan",
            "search_tools",
            "list_skills",
            "load_skill",
            "load_tool",
            "ask_user",
            "remember_preference",
        ]
    }


def test_second_turn_reuses_session_history() -> None:
    provider = FakeProvider(["answer"])
    client = make_client(provider)
    first = client.post("/v1/chat/stream", json={"message": "first"})
    session_id = session_id_from(first.text)

    second = client.post(
        "/v1/chat/stream",
        json={"session_id": session_id, "message": "second"},
    )

    assert second.status_code == 200
    assert [message.content for message in provider.requests[-1]] == [
        "first",
        "answer",
        "second",
    ]


def test_tool_call_streams_draft_then_resets_it_and_never_leaks_it_into_the_answer() -> (
    None
):
    adapter = QuoteAdapter()
    provider = ScriptedProvider(
        [
            tool_round(
                ToolUse("call_1", "get_quote", {"symbol": "600519"}),
                draft=("草稿：", "查询中"),
            ),
            text_round("贵州茅台最新报价 100.0"),
        ]
    )
    client = make_client(provider, data_access=DataAccess([adapter]))

    response = client.post("/v1/chat/stream", json={"message": "贵州茅台报价"})

    events = parse_events(response.text)
    names = [name for name, _ in events]
    statuses = [payload["status"] for name, payload in events if name == "tool_status"]
    streamed = "".join(payload["text"] for name, payload in events if name == "delta")
    answers = [payload["text"] for name, payload in events if name == "answer"]
    reset_at = names.index("text_reset")
    after_reset = "".join(
        payload["text"] for name, payload in events[reset_at + 1 :] if name == "delta"
    )
    assert statuses == ["started", "completed"]
    # The draft streams live, then text_reset clears it before the answer streams.
    assert "草稿" in streamed
    assert "草稿" not in after_reset
    assert "草稿" not in "".join(answers)
    assert answers == ["贵州茅台最新报价 100.0"]
    assert adapter.quoted == ["600519"]
    assert names[-1] == "done"
    assert events[-1][1]["tool_calls"] == 1
    # The last outgoing message is the research-state view; the tool_result is
    # the last real history entry before it (docs 3.3).
    roles = [message.role for message in provider.requests[-1]]
    assert "tool_result" in roles


def test_session_loop_receives_the_default_system_prompt() -> None:
    provider = ScriptedProvider([text_round("ok")])
    client = make_client(provider)

    response = client.post("/v1/chat/stream", json={"message": "hi"})

    assert response.status_code == 200
    assert provider.systems == [DEFAULT_SYSTEM_PROMPT]


def test_cancelling_the_stream_cancels_the_loop_task_and_frees_the_session(
    tmp_path,
) -> None:
    async def run():
        data = BlockingQuoteData()
        provider = ScriptedProvider(
            [
                tool_round(
                    ToolUse("call_1", "get_quote", {"symbol": "600519"}),
                    draft=("草稿",),
                ),
                text_round("迟到的答案"),
            ]
        )
        asgi_app = create_app(
            provider,
            data_access=data,
            settings=Settings(
                data={"cache_dir": tmp_path / "cache"},
                paths={
                    "output_dir": tmp_path / "output",
                    "memory_db": tmp_path / "cache" / "memory.db",
                },
            ),
        )
        stream = AsgiStream(asgi_app, {"message": "查询贵州茅台"})
        task = stream.start()
        await stream.wait_for('"status": "started"')
        session_id = session_id_from(stream.text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: data.cancelled)

        second = AsgiStream(asgi_app, {"session_id": session_id, "message": "再问一次"})
        second_task = second.start()
        await second.wait_for("event: done")
        await second_task
        return stream, second, session_id, provider

    stream, second, session_id, provider = asyncio.run(run())

    assert "event: done" not in stream.text()
    assert "event: answer" not in stream.text()
    assert f'"session_id": "{session_id}"' in second.text()
    assert "event: answer" in second.text()
    follow_up = provider.requests[-1]
    requested = {
        tool_use.call_id for message in follow_up for tool_use in message.tool_uses
    }
    answered = {
        call_id
        for message in follow_up
        if message.role == "tool_result"
        for call_id, _ in message.tool_results
    }
    assert requested == answered == {"call_1"}


def test_production_factory_runs_runtime_audit_validation(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "model": {"provider": "fake"},
                "audit": {"log_path": "missing/audit.jsonl"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="审计日志父目录不可写"):
        create_production_app(settings_path)


def test_production_factory_starts_without_an_api_key(monkeypatch, tmp_path) -> None:
    """Startup must not require credentials; the UI configures the provider later."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {"model": {"provider": "deepseek"}, "audit": {"log_path": "audit.jsonl"}}
        ),
        encoding="utf-8",
    )

    app = create_production_app(settings_path)

    assert app is not None
    assert app.title == "FinHarness"
