"""`POST /v1/chat/stop`：协作式停止（docs 03.3）。

覆盖端点的契约面，而不是引擎内部（那由 tests/engine/test_user_stop.py 覆盖）：

* 对在飞的生成置位成功；
* 归属不符按 404（不泄露存在性），与其余按 id 取用的端点同一口径；
* 空闲时幂等返回 ``stopping: false``，停止一个已结束的生成不是错误；
* 认证不可绕过；
* 停止后对话历史仍可回放，并且 ``resumable`` 被报告出来，使刷新后仍能"继续"。

注意：get_current_user 优先读 cookie，其次才是 Bearer 头，而注册接口会下发
cookie。因此本文件里凡涉及身份的地方都先 ``cookies.clear()``，否则上一个注册
用户的 cookie 会盖过显式传入的令牌，把"他人会话"测成自己的。
"""

import asyncio
import threading

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.context.memory.store import MemoryStore
from finharness.provider.base import Provider
from finharness.server.api import create_app
from finharness.types import ModelUsage, Msg, StopSignal, StreamChunk, StreamEvent

from tests.server.conftest import authed_client, register_and_login


def _settings(tmp_path) -> Settings:
    return Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
    )


class HangingProvider(Provider):
    """只用于占位：本文件的停止断言都直接驱动注册表，不依赖真实流。"""

    async def stream(self, *, system, messages, tools, usage: ModelUsage):
        yield StreamChunk(StreamEvent.TEXT_DELTA, "开始回答")
        await asyncio.sleep(30)
        yield StreamChunk(StreamEvent.MESSAGE_END, ModelUsage())


def _app(tmp_path, provider=None):
    return create_app(provider=provider or HangingProvider(), settings=_settings(tmp_path))


def _register(client: TestClient, username: str) -> dict[str, str]:
    """注册一个用户并返回其显式 Bearer 头（cookie 由调用方负责清掉）。"""
    token, headers = register_and_login(client, username)
    client.cookies.clear()
    return headers


def _user_id(client: TestClient, headers: dict[str, str]) -> str:
    body = client.get("/v1/auth/me", headers=headers).json()
    return body["user"]["id"]


def test_stop_requires_auth(tmp_path):
    # 一个从未注册过的客户端：没有 cookie 也没有 Bearer，必须被拒绝。
    client = TestClient(_app(tmp_path))

    response = client.post("/v1/chat/stop", json={"session_id": "s_x"})

    assert response.status_code in (401, 403)


def test_stop_returns_404_for_unknown_session(tmp_path):
    client = authed_client(TestClient(_app(tmp_path)))

    response = client.post("/v1/chat/stop", json={"session_id": "s_does_not_exist"})

    assert response.status_code == 404


def test_stop_signals_a_busy_session(tmp_path):
    """端点对在飞的会话置位停止信号——这是整个功能的关键一步。"""
    client = authed_client(TestClient(_app(tmp_path)))
    registry = client.app.state.session_registry
    user_id = client.finharness_user["id"]

    async def prepare():
        session = await registry.ensure(None, conversation_id="c_busy", user_id=user_id)
        registry.mark_busy(session)
        session.stop_signal = StopSignal()
        return session

    session = asyncio.run(prepare())

    response = client.post(
        "/v1/chat/stop", json={"session_id": session.session_id}
    )

    assert response.status_code == 200
    assert response.json()["stopping"] is True
    assert session.stop_signal.requested is True


def test_stop_is_idempotent_when_session_is_idle(tmp_path):
    """会话存在但不在运行：返回 stopping:false，而不是错误。

    客户端无法（也不需要）区分"已经停了"与"本来就没在跑"，因此两者同解。
    """
    client = authed_client(TestClient(_app(tmp_path)))
    registry = client.app.state.session_registry
    user_id = client.finharness_user["id"]

    async def prepare():
        # 不 mark_busy：这是一个空闲窗口。
        return await registry.ensure(None, conversation_id="c_idle", user_id=user_id)

    asyncio.run(prepare())

    response = client.post("/v1/chat/stop", json={"conversation_id": "c_idle"})

    assert response.status_code == 200
    assert response.json()["stopping"] is False


def test_stop_does_not_leak_other_users_sessions(tmp_path):
    """他人的 session_id 视同不存在：归属校验按 404 处理，不泄露存在性。"""
    client = TestClient(_app(tmp_path))
    headers_a = _register(client, "user_a")
    headers_b = _register(client, "user_b")
    registry = client.app.state.session_registry

    async def prepare():
        session = await registry.ensure(
            None, conversation_id="c_owned_by_a", user_id=_user_id(client, headers_a)
        )
        registry.mark_busy(session)
        session.stop_signal = StopSignal()
        return session

    session = asyncio.run(prepare())

    response = client.post(
        "/v1/chat/stop", json={"session_id": session.session_id}, headers=headers_b
    )

    assert response.status_code == 404
    # 信号没被置位：他人的请求不得影响这个会话。
    assert session.stop_signal.requested is False


def test_resumable_is_reported_after_a_stopped_turn(tmp_path):
    """被停止后，历史回放要报告 resumable，使刷新后仍能"继续研究"。"""
    settings = _settings(tmp_path)
    store = MemoryStore(settings.paths.memory_db)
    client = authed_client(TestClient(create_app(provider=HangingProvider(), settings=settings)))
    user_id = client.finharness_user["id"]
    store.ensure_conversation("c_resume", user_id=user_id, title="t")
    store.append_messages("c_resume", [Msg.user("被停止的问题")])
    store.save_checkpoint(
        "c_resume", status="stopped", reason="user_stopped", rounds=2, partial_answer="部分结论"
    )

    body = client.get("/v1/conversations/c_resume/messages").json()

    assert body["resumable"] is not None
    assert body["resumable"]["reason"] == "user_stopped"
    assert body["resumable"]["rounds"] == 2


def test_resumable_is_null_after_a_completed_turn(tmp_path):
    settings = _settings(tmp_path)
    store = MemoryStore(settings.paths.memory_db)
    client = authed_client(TestClient(create_app(provider=HangingProvider(), settings=settings)))
    user_id = client.finharness_user["id"]
    store.ensure_conversation("c_done", user_id=user_id, title="t")
    store.append_messages("c_done", [Msg.user("正常问题"), Msg(role="assistant", content="答案")])
    store.save_checkpoint("c_done", status="completed")

    body = client.get("/v1/conversations/c_done/messages").json()

    assert body["resumable"] is None


def test_stream_stops_cleanly_when_the_endpoint_is_hit(tmp_path):
    """端到端：真流式请求在运行中被 /v1/chat/stop 命中后干净收尾。

    这是唯一一条把"端点置位 -> 引擎观察到 -> 终止帧下发"串起来的测试，
    因此它断言的是完整契约：有 done、无 error、reason 为 user_stopped。
    """
    settings = _settings(tmp_path)
    client = authed_client(
        TestClient(create_app(provider=SlowStreamProvider(), settings=settings))
    )

    result: dict = {}

    def consume() -> None:
        response = client.post(
            "/v1/chat/stream",
            json={"message": "一个长问题", "conversation_id": "c_e2e"},
        )
        result["text"] = response.text
        result["status"] = response.status_code

    worker = threading.Thread(target=consume)
    worker.start()
    # 等引擎确实开始流式输出，再发停止；否则可能停在一个尚未 mark_busy 的会话上。
    assert SlowStreamProvider.started.wait(timeout=10), "provider never started"
    stop = client.post("/v1/chat/stop", json={"conversation_id": "c_e2e"})
    worker.join(timeout=15)

    assert stop.status_code == 200
    assert stop.json()["stopping"] is True
    assert not worker.is_alive(), "stream did not terminate after stop"
    assert result["status"] == 200
    assert "event: done" in result["text"]
    # 停止不是失败，因此不该有任何 error 帧。
    assert "event: error" not in result["text"]
    assert "user_stopped" in result["text"]


class SlowStreamProvider(Provider):
    """持续流式输出直到被停止：给端到端测试一个足够长的窗口。

    ``started`` 与 ``stop_requested`` 是类级信号，使并发发起的停止请求
    （另一个线程）能观察到同一个状态。
    """

    started = threading.Event()
    stop_requested = threading.Event()

    async def stream(self, *, system, messages, tools, usage: ModelUsage):
        for index in range(200):
            yield StreamChunk(StreamEvent.TEXT_DELTA, f"第{index}段 ")
            if index == 0:
                self.__class__.started.set()
            await asyncio.sleep(0.05)
        yield StreamChunk(StreamEvent.MESSAGE_END, ModelUsage())
