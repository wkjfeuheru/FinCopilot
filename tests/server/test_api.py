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
    """离线 provider 替身，按请求回放预设的轮次。"""

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
    """永不返回的行情来源，以便能观察到取消行为。"""

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
    """以一个可取消的 task 驱动单个 ASGI 请求，并记录 SSE 帧。

    该通道绕过 TestClient 直连 ASGI，因此认证头必须显式传入。
    """

    def __init__(self, asgi_app, payload: dict, token: str = "") -> None:
        self.asgi_app = asgi_app
        self.payload = payload
        self.token = token
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
        headers = [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
        ]
        if self.token:
            headers.append((b"authorization", f"Bearer {self.token}".encode()))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/chat/stream",
            "raw_path": b"/v1/chat/stream",
            "query_string": b"",
            "headers": headers,
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
    """构建一个客户端，其 memory store 与缓存位于临时目录中。

    若不显式传入 settings 对象，应用会使用仓库中的
    data_cache/memory.db，于是每个聊天测试都会把对话追加到
    开发者真实的 store 中。
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
            "auth_db": tmp_path / "cache" / "users.db",
        },
    )
    from tests.server.conftest import authed_client

    client = TestClient(create_app(provider, data_access=data_access, settings=settings))
    return authed_client(client)


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
    # 始终报告缓存拆分，以便调用方计算命中率。
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
            "get_research_reports",
            "get_macro_indicators",
            "get_industry_perf",
            "get_industry_constituents",
            "calc_metrics",
            "calc_valuation",
            "run_backtest",
            "make_chart",
            "write_report",
            "read_file",
            "write_file",
            "web_search",
            "research_plan",
            "update_plan_step",
            "record_conclusion",
            "search_tools",
            "list_skills",
            "load_skill",
            "load_tool",
            "spawn_agent",
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
    # 草稿会实时流式输出，随后 text_reset 在答案开始流式输出前将其清除。
    assert "草稿" in streamed
    assert "草稿" not in after_reset
    assert "草稿" not in "".join(answers)
    assert answers == ["贵州茅台最新报价 100.0"]
    assert adapter.quoted == ["600519"]
    assert names[-1] == "done"
    assert events[-1][1]["tool_calls"] == 1
    # 最后一条发往模型的消息是研究状态视图；tool_result 是
    # 它之前最后一条真实的历史条目（docs 3.3）。
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
                    "auth_db": tmp_path / "cache" / "users.db",
                },
            ),
        )
        # 直连 ASGI 的通道不走 TestClient，因此先注册一个用户取得令牌。
        from tests.server.conftest import register_and_login

        token, _ = register_and_login(TestClient(asgi_app))
        stream = AsgiStream(asgi_app, {"message": "查询贵州茅台"}, token=token)
        task = stream.start()
        await stream.wait_for('"status": "started"')
        session_id = session_id_from(stream.text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: data.cancelled)

        second = AsgiStream(
            asgi_app, {"session_id": session_id, "message": "再问一次"}, token=token
        )
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
    """启动时不得要求凭据；provider 由 UI 稍后配置。"""
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


def test_metrics_endpoint_is_absent_when_disabled(tmp_path) -> None:
    """默认关闭 metrics 时不注册 /metrics，避免暴露一个空端点。"""
    settings = Settings(
        model={"provider": "fake"},
        data={"cache_dir": tmp_path / "cache"},
        paths={"output_dir": tmp_path / "out", "memory_db": tmp_path / "cache" / "memory.db"},
    )
    client = TestClient(create_app(FakeProvider(["hi"]), settings=settings))

    assert client.get("/metrics").status_code == 404


def test_metrics_endpoint_reports_request_and_token_counters(tmp_path) -> None:
    """打开 metrics 后，一次对话应产出请求计数与 token 计数。"""
    settings = Settings(
        model={"provider": "fake"},
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "out",
            "memory_db": tmp_path / "cache" / "memory.db",
            "auth_db": tmp_path / "cache" / "users.db",
        },
        observability={"metrics": {"enabled": True}},
    )
    from tests.server.conftest import authed_client

    client = authed_client(TestClient(create_app(FakeProvider(["hello", " world"]), settings=settings)))

    with client.stream("POST", "/v1/chat/stream", json={"message": "question"}) as response:
        for _ in response.iter_lines():
            pass

    body = client.get("/metrics").text
    assert "agent_request_duration_seconds_count" in body
    assert 'llm_tokens_total{call_type="main",kind="input"' in body
    assert "model=" in body


def test_production_factory_rejects_metrics_without_the_optional_dependency(
    monkeypatch, tmp_path
) -> None:
    """开关打开却没装依赖时必须启动即失败，而不是静默不采集。"""
    import finharness.config.settings as settings_module

    monkeypatch.setattr(settings_module, "_module_available", lambda name: False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "model": {"provider": "fake"},
                "audit": {"log_path": "audit.jsonl"},
                "observability": {"metrics": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="prometheus-client"):
        create_production_app(settings_path)


def test_production_factory_rejects_tracing_without_a_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "model": {"provider": "fake"},
                "audit": {"log_path": "audit.jsonl"},
                "observability": {"tracing": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="LANGSMITH_API_KEY"):
        create_production_app(settings_path)
