"""交互式通道：引擎发出通知，客户端作答，回合继续。

在引擎边界（真实的 ConfirmBus 加 AskUserTool）和 HTTP 边界
（``POST /v1/chat/respond`` 解决挂起的请求）进行验证。浏览器对话框
本身由人工验收覆盖，不在这批测试范围内。
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.server.api import create_app
from finharness.server.confirm import ConfirmBus
from finharness.tools.meta.ask import AskUserTool
from tests.server.conftest import authed_client


def _settings(tmp_path) -> Settings:
    return Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
    )


async def _answer_latest(bus: ConfirmBus, value: str) -> None:
    """像客户端那样，在下一个 tick 回答刚刚注册的请求。"""
    await asyncio.sleep(0)
    pending = bus.pending_ids()
    if pending:
        bus.respond(request_id=pending[-1], value=value)


def test_ask_user_reports_the_answer_it_receives():
    bus = ConfirmBus(ttl_s=2.0)
    data = DataAccess([], settings=Settings())

    async def run():
        async def bridge(kind, prompt, options, **_):
            _, answer = await bus.request(
                session_id="s", kind=kind, prompt=prompt, options=options,
                announce=lambda payload: _answer_latest(bus, "近一年"),
            )
            return answer

        tool = AskUserTool(data)
        tool.interactive = bridge
        return await tool.run(question="你关注哪个时间段？", options=["近一年", "近三年"])

    result = asyncio.run(run())

    assert result.ok is True
    assert "近一年" in result.content


def test_ask_user_announces_multi_select_and_joins_answers():
    """多选：multi_select 随 payload 下发；以「；」连接的答案原样回传模型。"""
    bus = ConfirmBus(ttl_s=2.0)
    announced: list[dict] = []
    data = DataAccess([], settings=Settings())

    async def run():
        async def announce(payload):
            announced.append(payload)
            await _answer_latest(bus, "近一年；近三年")

        async def bridge(kind, prompt, options, *, multi_select=False):
            _, answer = await bus.request(
                session_id="s", kind=kind, prompt=prompt, options=options,
                multi_select=multi_select, announce=announce,
            )
            return answer

        tool = AskUserTool(data)
        tool.interactive = bridge
        return await tool.run(
            question="你关注哪些时间段？", options=["近一年", "近三年"], multi_select=True
        )

    result = asyncio.run(run())

    assert announced and announced[0]["multi_select"] is True
    assert result.ok is True
    assert "近一年；近三年" in result.content


def test_ask_user_announces_the_request_before_waiting():
    """必须把该请求告知客户端，而不只是告知答案。"""
    bus = ConfirmBus(ttl_s=2.0)
    announced: list[dict] = []

    async def run():
        async def announce(payload):
            announced.append(payload)
            await _answer_latest(bus, "y")

        await bus.request(
            session_id="s", kind="confirm", prompt="run?", options=["y", "n"],
            announce=announce,
        )

    asyncio.run(run())

    assert announced and announced[0]["kind"] == "confirm"
    assert announced[0]["prompt"] == "run?"
    assert announced[0]["options"] == ["y", "n"]
    assert announced[0]["request_id"]


def test_ask_user_reports_a_timeout_when_unanswered():
    data = DataAccess([], settings=Settings())

    async def run():
        tool = AskUserTool(data)

        async def never_answered(kind, prompt, options, **_):
            await asyncio.sleep(0.05)

        tool.interactive = never_answered
        return await tool.run(question="?")

    result = asyncio.run(run())

    assert result.ok is True
    # 必须让模型知道用户从未作答，而不是收到一个编造的答案。
    assert "未在时限内回答" in result.content


def test_ask_user_without_a_channel_fails_cleanly():
    data = DataAccess([], settings=Settings())

    result = asyncio.run(AskUserTool(data).run(question="?"))

    assert result.ok is False
    assert "不支持交互提问" in result.error


def test_respond_endpoint_rejects_an_unknown_request(tmp_path):
    client = authed_client(TestClient(create_app(provider=None, settings=_settings(tmp_path))))

    response = client.post("/v1/chat/respond", json={"request_id": "req_nope", "response": "y"})

    assert response.status_code == 404


def test_respond_endpoint_resolves_the_apps_pending_request(tmp_path):
    """通过真实的 HTTP 回答注册在应用自身总线上的请求。"""
    app = create_app(provider=None, settings=_settings(tmp_path))
    client = authed_client(TestClient(app))
    bus = app.state.confirm_bus

    async def scenario():
        async def answer_via_http():
            await asyncio.sleep(0.01)
            pending = bus.pending_ids()
            assert pending, "request should be registered"
            reply = client.post(
                "/v1/chat/respond", json={"request_id": pending[-1], "response": "y"}
            )
            assert reply.status_code == 200

        task = asyncio.create_task(answer_via_http())
        _, answer = await bus.request(session_id="s", kind="confirm", prompt="?")
        await task
        return answer

    assert asyncio.run(scenario()) == "y"


# -- FSM-owned confirmation ordering (Task 6) ---------------------------------


class _ConfirmBusPort:
    """InteractivePort wrapping ConfirmBus; emits interactive_request via sink."""

    def __init__(self, bus: ConfirmBus, sink, *, answer: str = "y") -> None:
        self.bus = bus
        self.sink = sink
        self.answer = answer
        self.request_ids: list[str] = []

    async def prompt(self, spec) -> str | None:
        from finharness.types import EngineEvent

        async def announce(payload: dict) -> None:
            self.request_ids.append(payload["request_id"])
            await self.sink.emit(EngineEvent("interactive_request", payload))
            await _answer_latest(self.bus, self.answer)

        kind = "question" if getattr(spec, "kind", "") == "question" else "confirm"
        payload, answer = await self.bus.request(
            session_id="s",
            kind=kind,
            prompt=spec.prompt,
            options=list(spec.options),
            multi_select=bool(getattr(spec, "multi_select", False)),
            announce=announce,
        )
        await self.sink.emit(
            EngineEvent(
                "interaction_resolved",
                {
                    "request_id": payload.get("request_id"),
                    "answer": answer,
                    "timeout": answer is None,
                },
            )
        )
        return answer


async def run_confirming_tool(tmp_path, answer: str = "y"):
    """Drive a write tool that needs confirmation; return engine events."""
    from test_loop import ScriptedProvider, Sink, StubRegistry, text_round, tool_round

    from finharness.config.settings import ContextSettings, PermissionSettings, ToolSettings
    from finharness.context.memory.store import MemoryStore
    from finharness.engine.loop import AgentLoop
    from finharness.permissions.gate import PermissionGate
    from finharness.tools.base import PermissionLevel
    from finharness.types import ToolResult, ToolUse
    from tests.conftest import settings_with_cache

    class WriteTool:
        name = "write"
        permission = PermissionLevel.WRITE
        timeout = None

        async def run(self, **kwargs) -> ToolResult:
            return ToolResult(content="wrote", ok=True)

    settings = settings_with_cache(
        tmp_path,
        permission=PermissionSettings(default_mode="default"),
        context=ContextSettings(max_turns=30, max_result_tokens=1000),
        tools=ToolSettings(timeout_default_s=30),
    )
    store = MemoryStore(tmp_path / "memory.db")
    sink = Sink()
    bus = ConfirmBus(ttl_s=2.0)
    port = _ConfirmBusPort(bus, sink, answer=answer)
    gate = PermissionGate(settings=settings)
    tool = WriteTool()
    registry = StubRegistry({"write": tool}, read_only=set())
    provider = ScriptedProvider(
        [
            tool_round(ToolUse("c1", "write", {"path": "out.txt"})),
            text_round("done"),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        registry=registry,
        settings=settings,
        system="test",
        output=sink,
        store=store,
        conversation_id="conv",
        user_id="",
        gate=gate,
        interactive=port,
    )
    await loop.run("please write")
    return sink.events


def test_confirmation_state_precedes_interactive_request(tmp_path):
    events = asyncio.run(run_confirming_tool(tmp_path, answer="y"))
    kinds = [event.kind for event in events]
    awaiting = next(
        i
        for i, event in enumerate(events)
        if event.kind == "state" and event.data["phase"] == "awaitingconfirmation"
    )
    request = kinds.index("interactive_request")
    assert awaiting < request


# -- 网络外发确认（docs 03.7.1）------------------------------------------------


def test_egress_confirmation_offers_remember_option(tmp_path):
    """网络外发确认必须给用户"本对话不再询问"，而不是只有一次性的 y/n。"""
    from finharness.permissions.gate import PermissionGate
    from tests.permissions.test_gate import FakeTool, make_settings

    bus = ConfirmBus(ttl_s=2.0)
    settings = make_settings(tmp_path)
    confirmed: set[str] = set()
    announced: list[dict] = []

    async def confirm_egress(name, args):
        async def announce(payload):
            announced.append(payload)
            await _answer_latest(bus, "y_remember")

        _, answer = await bus.request(
            session_id="s",
            kind="confirm",
            prompt=f"{name} 将访问外部网络",
            options=["y", "y_remember", "n"],
            announce=announce,
        )
        if answer == "y_remember":
            confirmed.add("egress")
            return True
        return answer == "y"

    gate = PermissionGate(
        settings=settings,
        confirm_egress=confirm_egress,
        conversation_id="conv-1",
        confirmed_categories=confirmed,
    )

    async def scenario():
        return await gate.check(FakeTool(name="web_search", egress=True), {"query": "政策"})

    decision = asyncio.run(scenario())

    assert decision.verdict.value == "allow"
    assert announced[0]["options"] == ["y", "y_remember", "n"]
    assert "egress" in confirmed


def test_egress_remember_suppresses_the_next_prompt(tmp_path):
    """记住之后，同一对话的第二次外发不再宣告确认请求。"""
    from finharness.permissions.gate import PermissionGate
    from tests.permissions.test_gate import FakeTool, make_settings

    settings = make_settings(tmp_path)
    confirmed: set[str] = set()
    prompts = 0

    async def confirm_egress(name, args):
        nonlocal prompts
        prompts += 1
        return True

    gate = PermissionGate(
        settings=settings,
        confirm_egress=confirm_egress,
        conversation_id="conv-1",
        confirmed_categories=confirmed,
    )

    async def scenario():
        # 首次确认（模拟"允许并本对话不再询问"）
        first = await gate.check(FakeTool(name="web_search", egress=True), {"query": "a"})
        confirmed.add("egress")
        second = await gate.check(FakeTool(name="web_search", egress=True), {"query": "b"})
        return first, second

    first, second = asyncio.run(scenario())

    assert first.verdict.value == "allow"
    assert second.verdict.value == "allow"
    assert "本对话已确认" in second.reason
    assert prompts == 1  # 第二次没有再次询问


# -- 并发同类请求的合并（docs 03.7.1）------------------------------------------


def test_dedupe_merges_concurrent_requests_into_one_prompt():
    """同 key 的并发请求只宣告一次，且共享同一个 request_id 与答案。"""
    bus = ConfirmBus(ttl_s=2.0)
    announced: list[dict] = []

    async def run():
        async def ask():
            async def announce(payload):
                announced.append(payload)

            return await bus.request(
                session_id="s",
                kind="confirm",
                prompt="联网?",
                options=["y", "n"],
                announce=announce,
                dedupe_key="egress:conv-1",
            )

        first = asyncio.ensure_future(ask())
        await asyncio.sleep(0)
        second = asyncio.ensure_future(ask())
        await asyncio.sleep(0)
        # 两个等待者共享一条登记，因此只会有一份提示。
        assert len(bus.pending_ids()) == 1
        ids = bus.pending_ids()
        assert bus.respond(request_id=ids[0], value="y") is True
        return await first, await second

    (first_payload, first_answer), (second_payload, second_answer) = asyncio.run(run())

    assert len(announced) == 1
    assert first_payload["request_id"] == second_payload["request_id"]
    assert first_answer == "y"
    assert second_answer == "y"
    assert bus.pending_ids() == []


def test_dedupe_timeout_cleans_up_and_reasks_fresh():
    """超时后不得留下悬空条目：同 key 的后续请求必须拿到新 id 再问一次。"""
    bus = ConfirmBus(ttl_s=0.05)

    async def run():
        async def ask():
            return await bus.request(
                session_id="s", kind="confirm", prompt="?", dedupe_key="egress:conv-1"
            )

        first = await ask()
        assert bus.pending_ids() == []
        second = await ask()
        return first, second

    (first_payload, first_answer), (second_payload, second_answer) = asyncio.run(run())

    assert first_answer is None
    assert second_answer is None
    assert first_payload["request_id"] != second_payload["request_id"]


def test_cancel_session_fails_every_shared_waiter():
    """停止/断线要唤醒共享同一 future 的所有等待者，而不是只唤醒一个。"""
    bus = ConfirmBus(ttl_s=5.0)

    async def run():
        async def ask():
            return await bus.request(
                session_id="s", kind="confirm", prompt="?", dedupe_key="egress:conv-1"
            )

        tasks = [asyncio.ensure_future(ask()) for _ in range(3)]
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(bus.pending_ids()) == 1
        assert bus.cancel_session("s") == 1
        for task in tasks:
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())

    assert bus.pending_ids() == []


def test_dedupe_does_not_hand_a_resolved_answer_to_a_new_caller():
    """落定后的新调用者必须重新提问，不能白拿上一次的答案。

    否则一次性的 "y" 授权会被该类别后续的调用反复复用，用户只被问了一次
    却放行了整个对话。
    """
    bus = ConfirmBus(ttl_s=2.0)
    announced: list[str] = []

    async def run():
        async def ask():
            async def announce(payload):
                announced.append(payload["request_id"])

            return await bus.request(
                session_id="s",
                kind="confirm",
                prompt="?",
                announce=announce,
                dedupe_key="egress:conv-1",
            )

        first = asyncio.ensure_future(ask())
        await asyncio.sleep(0)
        bus.respond(request_id=bus.pending_ids()[0], value="y")
        first_id = (await first)[0]["request_id"]

        # 旧的共享条目已落定但可能尚未释放：新调用者仍须拿到新 id 并再问一次。
        second = asyncio.ensure_future(ask())
        for _ in range(3):
            await asyncio.sleep(0)
        pending = bus.pending_ids()
        assert len(pending) == 1 and pending[0] != first_id
        bus.respond(request_id=pending[0], value="n")
        return first_id, (await second)[0]["request_id"]

    first_id, second_id = asyncio.run(run())

    assert first_id != second_id
    assert announced == [first_id, second_id]


def test_concurrent_egress_checks_share_one_confirmation(tmp_path):
    """同一轮并发的多个 web_search 只向用户问一次，一次授权放行全部。"""
    from finharness.permissions.gate import PermissionGate
    from tests.permissions.test_gate import FakeTool, make_settings

    bus = ConfirmBus(ttl_s=2.0)
    settings = make_settings(tmp_path)
    confirmed: set[str] = set()
    announced: list[dict] = []

    async def confirm_egress(name, args):
        async def announce(payload):
            announced.append(payload)

        _, answer = await bus.request(
            session_id="s",
            kind="confirm",
            prompt=f"{name} 将访问外部网络",
            options=["y", "y_remember", "n"],
            announce=announce,
            dedupe_key="egress:conv-1",
        )
        if answer == "y_remember":
            confirmed.add("egress")
            return True
        return answer == "y"

    gate = PermissionGate(
        settings=settings,
        confirm_egress=confirm_egress,
        conversation_id="conv-1",
        confirmed_categories=confirmed,
    )

    async def scenario():
        tasks = [
            asyncio.ensure_future(gate.check(FakeTool(name="web_search", egress=True), {"query": q}))
            for q in ("a", "b", "c")
        ]
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(bus.pending_ids()) == 1
        bus.respond(request_id=bus.pending_ids()[0], value="y_remember")
        return await asyncio.gather(*tasks)

    decisions = asyncio.run(scenario())

    assert len(announced) == 1
    assert all(decision.verdict.value == "allow" for decision in decisions)
    assert "egress" in confirmed
