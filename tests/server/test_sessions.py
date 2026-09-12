import asyncio

import pytest

from finharness.server.sessions import (
    SessionBusyError,
    SessionRegistry,
    new_conversation_id,
)


def factory(session_id=None, conversation_id=None):
    """Loop double: the registry hands it both ids."""
    return {"session_id": session_id, "conversation_id": conversation_id}


def test_session_registry_reuses_session_and_rejects_busy():
    async def run():
        # The factory receives the new session id so per-session state (such as
        # the citation registry) can be keyed to it.
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
    """Passing only a conversation id finds the open window for it."""
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
    """The conversation id survives; the execution window does not."""
    conversation_id = new_conversation_id()

    async def run():
        registry = SessionRegistry(factory, ttl_s=10)
        first = await registry.ensure(None, conversation_id=conversation_id)
        first.last_active -= 60  # age it past the TTL
        second = await registry.ensure(None, conversation_id=conversation_id)
        return first, second

    first, second = asyncio.run(run())

    assert first.session_id != second.session_id
    assert second.conversation_id == conversation_id


def test_expired_sessions_are_evicted():
    """The registry must not grow without bound.

    Previously an expired entry stayed in the dict forever, holding its loop and
    transcript; now that conversations persist in the memory store, releasing the
    in-memory session costs nothing.
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
    """A session mid-request must survive its TTL until it finishes."""

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
