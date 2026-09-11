import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from finharness.config.settings import SettingsError
from finharness.data.access import DataAccess, RawData
from finharness.data.adapters.base import DataAdapter
from finharness.provider.base import Provider
from finharness.provider.fake import FakeProvider
from finharness.server.api import DEFAULT_SYSTEM_PROMPT, app, create_app, create_production_app
from finharness.types import ModelUsage, StreamChunk, StreamEvent, ToolUse


class ScriptedProvider(Provider):
    """Offline provider double replaying canned rounds per request."""

    def __init__(self, rounds: list[list[StreamChunk] | Exception]):
        self.rounds = list(rounds)
        self.requests: list[list] = []
        self.systems: list[str] = []

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
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

        self.quoted.append(symbol)
        return pd.DataFrame([{"symbol": symbol, "price": 100.0}])


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


def message_end(*tool_uses: ToolUse, input_tokens: int = 1, output_tokens: int = 1) -> StreamChunk:
    return StreamChunk(
        StreamEvent.MESSAGE_END,
        ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens, tool_uses=list(tool_uses)),
    )


def text_round(*parts: str) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in parts] + [message_end()]


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
            "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
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


def test_health_endpoint_returns_service_status() -> None:
    response = TestClient(app).get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_stream_returns_session_and_answer_events() -> None:
    client = TestClient(create_app(FakeProvider(["hello", " world"])))

    response = client.post("/v1/chat/stream", json={"message": "question"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: session" in response.text
    assert 'event: delta\ndata: {"text": "hello"}' in response.text
    assert 'event: answer\ndata: {"text": "hello world"}' in response.text
    assert "event: done" in response.text


def test_done_event_carries_the_session_id_and_usage() -> None:
    client = TestClient(create_app(FakeProvider(["hello"])))

    response = client.post("/v1/chat/stream", json={"message": "question"})

    events = parse_events(response.text)
    names = [name for name, _ in events]
    done = events[-1]
    assert names.count("done") == 1
    assert done[0] == "done"
    assert done[1]["session_id"] == session_id_from(response.text)
    assert done[1]["succeeded"] is True
    assert done[1]["usage"] == {"input_tokens": 1, "output_tokens": 1}
    assert done[1]["cost_cny"] == 0.0
    assert done[1]["tool_calls"] == 0


def test_chat_stream_reports_provider_error_and_done_as_sse_events() -> None:
    client = TestClient(create_app(FakeProvider([], error=RuntimeError("provider down"))))

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


def test_tools_endpoint_lists_m0_financial_tools() -> None:
    client = TestClient(create_app(FakeProvider(["ok"])))

    response = client.get("/v1/tools")

    assert response.json() == {"tools": ["get_quote", "get_kline", "get_indicators"]}


def test_second_turn_reuses_session_history() -> None:
    provider = FakeProvider(["answer"])
    client = TestClient(create_app(provider))
    first = client.post("/v1/chat/stream", json={"message": "first"})
    session_id = session_id_from(first.text)

    second = client.post(
        "/v1/chat/stream",
        json={"session_id": session_id, "message": "second"},
    )

    assert second.status_code == 200
    assert [message.content for message in provider.requests[-1]] == ["first", "answer", "second"]


def test_tool_call_streams_tool_status_and_never_leaks_the_model_draft() -> None:
    adapter = QuoteAdapter()
    provider = ScriptedProvider(
        [
            tool_round(ToolUse("call_1", "get_quote", {"symbol": "600519"}), draft=("草稿：", "查询中")),
            text_round("贵州茅台最新报价 100.0"),
        ]
    )
    client = TestClient(create_app(provider, data_access=DataAccess([adapter])))

    response = client.post("/v1/chat/stream", json={"message": "贵州茅台报价"})

    events = parse_events(response.text)
    names = [name for name, _ in events]
    statuses = [payload["status"] for name, payload in events if name == "tool_status"]
    streamed = "".join(payload["text"] for name, payload in events if name == "delta")
    answers = [payload["text"] for name, payload in events if name == "answer"]
    assert statuses == ["started", "completed"]
    assert "草稿" not in streamed
    assert "草稿" not in "".join(answers)
    assert answers == ["贵州茅台最新报价 100.0"]
    assert adapter.quoted == ["600519"]
    assert names[-1] == "done"
    assert events[-1][1]["tool_calls"] == 1
    assert provider.requests[-1][-1].role == "tool_result"


def test_session_loop_receives_the_default_system_prompt() -> None:
    provider = ScriptedProvider([text_round("ok")])
    client = TestClient(create_app(provider))

    response = client.post("/v1/chat/stream", json={"message": "hi"})

    assert response.status_code == 200
    assert provider.systems == [DEFAULT_SYSTEM_PROMPT]


def test_cancelling_the_stream_cancels_the_loop_task_and_frees_the_session() -> None:
    async def run():
        data = BlockingQuoteData()
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("call_1", "get_quote", {"symbol": "600519"}), draft=("草稿",)),
                text_round("迟到的答案"),
            ]
        )
        asgi_app = create_app(provider, data_access=data)
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
        return stream, second, session_id

    stream, second, session_id = asyncio.run(run())

    assert "event: done" not in stream.text()
    assert "event: answer" not in stream.text()
    assert f'"session_id": "{session_id}"' in second.text()
    assert "event: answer" in second.text()


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
