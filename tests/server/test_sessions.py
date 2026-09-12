import asyncio

import pytest

from finharness.server.sessions import SessionBusyError, SessionRegistry


def test_session_registry_reuses_session_and_rejects_busy():
    async def run():
        # The factory receives the new session id so per-session state (such as
        # the citation registry) can be keyed to it.
        registry = SessionRegistry(lambda session_id: object())
        session = await registry.ensure(None)
        session.busy = True
        with pytest.raises(SessionBusyError):
            await registry.ensure(session.session_id)
        return await registry.ensure(None)

    new_session = asyncio.run(run())

    assert new_session.session_id.startswith("s_")


def test_expired_sessions_are_evicted():
    """The registry must not grow without bound.

    Previously an expired entry stayed in the dict forever, holding its loop and
    transcript; now that conversations persist in the memory store, releasing the
    in-memory session costs nothing.
    """

    async def run():
        registry = SessionRegistry(lambda session_id: object(), ttl_s=10)
        first = await registry.ensure(None)
        # Age the session past its TTL without it being busy.
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
        registry = SessionRegistry(lambda session_id: object(), ttl_s=10)
        session = await registry.ensure(None)
        session.busy = True
        session.last_active -= 60
        await registry.ensure(None)
        return registry, session

    registry, session = asyncio.run(run())

    assert session.session_id in registry.sessions
