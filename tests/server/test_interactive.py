"""交互式通道：引擎发出通知，客户端作答，回合继续。

在引擎边界（真实的 ConfirmBus 加 AskUserTool）和 HTTP 边界
（``POST /v1/chat/respond`` 解决挂起的请求）进行验证。浏览器对话框
本身由人工验收覆盖，不在这批测试范围内。
"""

import asyncio

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
            "memory_db": tmp_path / "cache" / "memory.db",
            "auth_db": tmp_path / "cache" / "users.db",
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
        async def bridge(kind, prompt, options):
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

        async def never_answered(kind, prompt, options):
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
