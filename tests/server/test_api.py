import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from finharness.auth.store import UserStore
from finharness.config.crypto import SecretCipher
from finharness.config.settings import Settings, SettingsError
from finharness.config.store import ConfigStore
from finharness.context.memory.store import MemoryStore
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
from finharness.types import (
    STATE_VIEW_META,
    ModelUsage,
    StreamChunk,
    StreamEvent,
    ToolUse,
)


def _is_state_view(message) -> bool:
    """随请求追加的会话研究状态视图，不是对话记录条目。"""
    return bool(message.metadata.get(STATE_VIEW_META))



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
        # 心跳是注释帧（以 ``:`` 开头，不含 event/data），按规范应被忽略。
        if frame.lstrip().startswith(":"):
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
    state/memory.db，于是每个聊天测试都会把对话追加到
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
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
    )
    from tests.server.conftest import authed_client

    client = TestClient(create_app(provider, data_access=data_access, settings=settings, single_tenant=True))
    return authed_client(client)


def test_health_endpoint_returns_service_status() -> None:
    response = TestClient(app).get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _ready_client(tmp_path) -> TestClient:
    audit_parent = tmp_path / "logs"
    audit_parent.mkdir()
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        audit={"log_path": audit_parent / "audit.jsonl"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
            "config_db": tmp_path / "state" / "config.db",
            "secret_key": tmp_path / "state" / "secret.key",
        },
    )
    return TestClient(create_app(settings=settings), raise_server_exceptions=False)


def test_ready_is_anonymous_when_persistent_stores_are_usable(tmp_path) -> None:
    response = _ready_client(tmp_path).get("/v1/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.parametrize("store_type", [UserStore, MemoryStore, ConfigStore])
def test_ready_returns_503_when_a_persistent_store_is_unavailable(
    tmp_path, monkeypatch, store_type
) -> None:
    def fail_ping(_store) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(store_type, "ping", fail_ping, raising=False)
    response = _ready_client(tmp_path).get("/v1/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_ready_returns_503_when_audit_parent_is_not_writable(
    tmp_path, monkeypatch
) -> None:
    client = _ready_client(tmp_path)
    monkeypatch.setattr("finharness.server.api.os.access", lambda *_args: False)

    response = client.get("/v1/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_ready_returns_503_when_audit_parent_check_raises(
    tmp_path, monkeypatch
) -> None:
    def fail_access(*_args) -> bool:
        raise OSError("filesystem unavailable")

    client = _ready_client(tmp_path)
    monkeypatch.setattr("finharness.server.api.os.access", fail_access)

    response = client.get("/v1/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_ready_returns_503_when_config_store_cannot_be_opened(
    tmp_path, monkeypatch
) -> None:
    class UnavailableConfigStore:
        def __init__(self, *_args, **_kwargs) -> None:
            raise OSError("database cannot be opened")

    monkeypatch.setattr("finharness.server.api.ConfigStore", UnavailableConfigStore)

    response = _ready_client(tmp_path).get("/v1/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def _trace_store_queries(monkeypatch, store) -> list[str]:
    queries: list[str] = []
    connect = store._connect

    def traced_connect():
        connection = connect()
        connection.set_trace_callback(queries.append)
        return connection

    monkeypatch.setattr(store, "_connect", traced_connect)
    return queries


def test_user_store_ping_uses_select_one_and_preserves_existing_users(
    tmp_path, monkeypatch
) -> None:
    store = UserStore(tmp_path / "users.db")
    issued = store.register("ping-user", "test-password")
    before = store.get_user(issued.user.id)
    queries = _trace_store_queries(monkeypatch, store)

    result = store.ping()

    assert result is None
    assert queries == ["SELECT 1"]
    assert store.count_users() == 1
    assert store.get_user(issued.user.id) == before


def test_memory_store_ping_uses_select_one_and_preserves_existing_conversations(
    tmp_path, monkeypatch
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    before = store.ensure_conversation("ping-conversation", user_id="ping-user")
    queries = _trace_store_queries(monkeypatch, store)

    result = store.ping()

    assert result is None
    assert queries == ["SELECT 1"]
    assert store.get_conversation("ping-conversation", user_id="ping-user") == before


def test_config_store_ping_uses_select_one_and_preserves_existing_provider_configs(
    tmp_path, monkeypatch
) -> None:
    store = ConfigStore(
        tmp_path / "config.db",
        cipher=SecretCipher(tmp_path / "secret.key"),
    )
    created = store.create(
        name="ping-provider",
        kind="openai_compat",
        base_url="https://example.test/v1",
        model="test-model",
        env_key=None,
        api_key="test-secret",
        activate=True,
        user_id="ping-user",
    )
    before = store.list_configs(user_id="ping-user")
    queries = _trace_store_queries(monkeypatch, store)

    result = store.ping()

    assert result is None
    assert queries == ["SELECT 1"]
    assert store.list_configs(user_id="ping-user") == before == [created]
    assert store.resolve_key(created.id, user_id="ping-user") == "test-secret"


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
            "list_fuyao_datasets",
            "query_a_share_data",
            "query_fund_data",
            "query_futures_data",
            "query_options_data",
            "calc_metrics",
            "calc_valuation",
            "run_backtest",
            "make_chart",
            "write_report",
            "read_file",
            "read_pdf",
            "write_file",
            "web_search",
            "research_plan",
            "update_plan_step",
            "record_conclusion",
            "search_tools",
            "spawn_agent",
            "summarize_document",
            "ask_user",
            "remember_preference",
            "search_memory",
            "update_memory",
            "forget_memory",
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
    # 末尾是随请求追加的研究状态视图（不是对话记录），故只看真实历史。
    assert [
        message.content for message in provider.requests[-1] if not _is_state_view(message)
    ] == [
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
            single_tenant=True,
            settings=Settings(
                data={"cache_dir": tmp_path / "cache"},
                paths={
                    "output_dir": tmp_path / "output",
                    "memory_db": tmp_path / "state" / "memory.db",
                    "auth_db": tmp_path / "state" / "users.db",
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


def test_production_factory_rejects_remote_insecure_cookie(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "model": {"provider": "fake"},
                "server": {"host": "0.0.0.0", "allow_remote": True},
                "auth": {"secure_cookie": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="auth.secure_cookie"):
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
        paths={"output_dir": tmp_path / "out", "memory_db": tmp_path / "state" / "memory.db"},
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
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
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


def test_stream_synthesizes_done_when_engine_ends_without_one(
    monkeypatch, tmp_path
) -> None:
    """引擎没有下发 ``done`` 就返回时，传输层必须补一个终止事件。

    否则浏览器只能看到连接被静默关闭，界面永远停在"运行中"，
    这正是"刷新后才看到结果"的成因。
    """
    from finharness.engine.loop import AgentLoop
    from finharness.types import AgentTurnOutcome

    async def silent_run(self, user_msg: str) -> AgentTurnOutcome:
        # 引擎正常返回，但不向 sink 发任何终止事件。
        return AgentTurnOutcome(answer="done-ish")

    monkeypatch.setattr(AgentLoop, "run", silent_run)
    client = make_client(FakeProvider(["unused"]), tmp_path=tmp_path)

    response = client.post("/v1/chat/stream", json={"message": "question"})

    events = parse_events(response.text)
    names = [name for name, _ in events]
    assert names[-1] == "done", names
    assert names.count("done") == 1
    assert events[-1][1]["succeeded"] is True


def test_stream_reports_terminal_error_when_engine_raises(monkeypatch, tmp_path) -> None:
    """引擎抛出异常时，客户端仍应收到 ``error`` 与 ``done``，而不是静默断流。"""
    from finharness.engine.loop import AgentLoop

    async def exploding_run(self, user_msg: str):
        raise RuntimeError("engine blew up")

    monkeypatch.setattr(AgentLoop, "run", exploding_run)
    client = make_client(FakeProvider(["unused"]), tmp_path=tmp_path)

    response = client.post("/v1/chat/stream", json={"message": "question"})

    events = parse_events(response.text)
    names = [name for name, _ in events]
    assert "error" in names, names
    assert names[-1] == "done", names
    error = next(payload for name, payload in events if name == "error")
    assert error["reason"] == "engine_error"
    assert events[-1][1]["succeeded"] is False


def test_stream_emits_heartbeat_during_silence(monkeypatch, tmp_path) -> None:
    """引擎长时间静默时应下发心跳注释帧，避免代理攒帧/截断连接。"""
    from finharness.engine.loop import AgentLoop
    from finharness.types import AgentTurnOutcome

    async def slow_run(self, user_msg: str) -> AgentTurnOutcome:
        await asyncio.sleep(0.3)
        return AgentTurnOutcome(answer="late")

    monkeypatch.setattr(AgentLoop, "run", slow_run)
    monkeypatch.setattr("finharness.server.api.HEARTBEAT_S", 0.05)
    client = make_client(FakeProvider(["unused"]), tmp_path=tmp_path)

    response = client.post("/v1/chat/stream", json={"message": "question"})

    assert ": keep-alive" in response.text
    # 心跳不得污染事件流：解析后仍只有终止性的 done 收尾。
    events = parse_events(response.text)
    assert [name for name, _ in events][-1] == "done"


def test_stream_response_sets_anti_buffering_headers(tmp_path) -> None:
    """SSE 响应必须禁止中间层缓存与缓冲，否则帧会被攒着一起发。"""
    client = make_client(FakeProvider(["hello"]), tmp_path=tmp_path)

    response = client.post("/v1/chat/stream", json={"message": "question"})

    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"


def test_chat_survives_unwritable_token_vocab_cache(tmp_path, monkeypatch) -> None:
    """词表缓存不可写不得把一轮对话打成 500（线上事故回归）。

    生产容器以非 root 运行、仓库根不可写，``TokenCounter`` 初始化里的 ``mkdir``
    抛 PermissionError，首次对话（AgentLoop 构造时）整个 500。契约是降级为字符
    近似，因此这里断言：即便词表缓存目录无法创建，本轮仍正常收尾于 ``done``。
    """
    # 父路径是文件 → 目录创建必然失败（Windows/Linux 一致的 OSError）。
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(blocker / "tiktoken"))

    # 共享计数器与默认目录都可能已被前序用例构造/环境变量化，重置到干净态。
    import finharness.context.tokens as tokens_mod

    monkeypatch.setattr(tokens_mod, "_SHARED_COUNTER", None)
    monkeypatch.setattr(
        tokens_mod, "VOCAB_CACHE_DIR", blocker / "tiktoken", raising=False
    )

    client = make_client(FakeProvider(["hello"]), tmp_path=tmp_path)
    response = client.post("/v1/chat/stream", json={"message": "question"})

    assert response.status_code == 200
    names = [name for name, _ in parse_events(response.text)]
    assert names[-1] == "done", response.text[:400]
