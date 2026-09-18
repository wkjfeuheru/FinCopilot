import asyncio

import pytest

from finharness.server.sessions import (
    SessionBusyError,
    SessionRegistry,
    new_conversation_id,
)


def factory(session_id=None, conversation_id=None, user_id=""):
    """循环替身：注册表把两个 id 与用户归属都交给它。"""
    return {"session_id": session_id, "conversation_id": conversation_id, "user_id": user_id}


def test_session_registry_reuses_session_and_rejects_busy():
    async def run():
        # 工厂接收新的 session id，这样按 session 隔离的状态（例如
        # 引用注册表）便能以其为键。
        registry = SessionRegistry(factory)
        session = await registry.ensure(None)
        session.busy = True
        with pytest.raises(SessionBusyError):
            await registry.ensure(session.session_id)
        return await registry.ensure(None)

    new_session = asyncio.run(run())

    assert new_session.session_id.startswith("s_")
    assert new_session.conversation_id.startswith("c_")


def test_new_session_gets_a_fresh_conversation_id():
    async def run():
        registry = SessionRegistry(factory)
        first = await registry.ensure(None)
        second = await registry.ensure(None)
        return first, second

    first, second = asyncio.run(run())

    assert first.conversation_id != second.conversation_id


def test_resuming_a_conversation_reuses_its_live_session():
    """只传入 conversation id，即可找到它对应的打开窗口。"""
    conversation_id = new_conversation_id()

    async def run():
        registry = SessionRegistry(factory)
        first = await registry.ensure(None, conversation_id=conversation_id)
        second = await registry.ensure(None, conversation_id=conversation_id)
        return first, second, registry

    first, second, registry = asyncio.run(run())

    assert first.session_id == second.session_id
    assert first.conversation_id == conversation_id
    assert len(registry.sessions) == 1


def test_resuming_an_expired_conversation_opens_a_new_window():
    """conversation id 会保留下来，执行窗口则不会。"""
    conversation_id = new_conversation_id()

    async def run():
        registry = SessionRegistry(factory, ttl_s=10)
        first = await registry.ensure(None, conversation_id=conversation_id)
        first.last_active -= 60  # 使其老化超过 TTL
        second = await registry.ensure(None, conversation_id=conversation_id)
        return first, second

    first, second = asyncio.run(run())

    assert first.session_id != second.session_id
    assert second.conversation_id == conversation_id


def test_expired_sessions_are_evicted():
    """注册表不能无界增长。

    此前过期的条目会永远留在 dict 中，占着它的循环与
    对话记录；如今对话持久化在 memory store 中，释放
    内存中的 session 不需要任何代价。
    """

    async def run():
        registry = SessionRegistry(factory, ttl_s=10)
        first = await registry.ensure(None)
        first.last_active -= 60
        second = await registry.ensure(None)
        return registry, first, second

    registry, first, second = asyncio.run(run())

    assert first.session_id not in registry.sessions
    assert second.session_id in registry.sessions
    assert len(registry.sessions) == 1


def test_busy_sessions_are_not_evicted():
    """处于请求处理中的 session 在完成之前必须挺过其 TTL。"""

    async def run():
        registry = SessionRegistry(factory, ttl_s=10)
        session = await registry.ensure(None)
        session.busy = True
        session.last_active -= 60
        await registry.ensure(None)
        return registry, session

    registry, session = asyncio.run(run())

    assert session.session_id in registry.sessions


def test_busy_conversation_cannot_be_resumed_concurrently():
    conversation_id = new_conversation_id()

    async def run():
        registry = SessionRegistry(factory)
        session = await registry.ensure(None, conversation_id=conversation_id)
        session.busy = True
        with pytest.raises(SessionBusyError):
            await registry.ensure(None, conversation_id=conversation_id)
        return session

    asyncio.run(run())


def test_abandoned_busy_session_is_eventually_reclaimed():
    """一个被遗弃的 busy 会话（流挂了、release 没跑到）不能永久占着内存。

    在飞请求不能被误伤，但 busy 持续到离谱就只能是被遗弃了。
    """

    async def run():
        registry = SessionRegistry(factory, ttl_s=10, busy_timeout_s=30)
        abandoned = await registry.ensure(None)
        registry.mark_busy(abandoned)
        # 模拟"很久以前就进入 busy 且从未释放"。
        abandoned.busy_since -= 3600
        await registry.ensure(None)
        return registry, abandoned

    registry, abandoned = asyncio.run(run())

    assert abandoned.session_id not in registry.sessions


def test_in_flight_busy_session_is_never_reclaimed():
    """刚开始的 busy 请求必须留着，否则同一对话会被并发写入。"""

    async def run():
        registry = SessionRegistry(factory, ttl_s=10, busy_timeout_s=30)
        session = await registry.ensure(None)
        registry.mark_busy(session)
        # 即便空闲时长超过 TTL，只要 busy 还"新"，就不能动它。
        session.last_active -= 3600
        await registry.ensure(None)
        return registry, session

    registry, session = asyncio.run(run())

    assert session.session_id in registry.sessions


def test_release_sweeps_expired_sessions():
    """一轮结束就该顺手回收其它过期会话，而不是等下一个请求。"""

    async def run():
        registry = SessionRegistry(factory, ttl_s=10)
        stale = await registry.ensure(None)
        active = await registry.ensure(None)
        # stale 空闲了很久，而 active 刚刚用过。
        stale.last_active -= 3600
        registry.release(active)
        return registry, stale, active

    registry, stale, active = asyncio.run(run())

    assert stale.session_id not in registry.sessions
    assert active.session_id in registry.sessions

