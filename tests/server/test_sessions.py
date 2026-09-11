import asyncio

import pytest

from finharness.server.sessions import SessionBusyError, SessionRegistry


def test_session_registry_reuses_session_and_rejects_busy():
    async def run():
        registry = SessionRegistry(lambda: object())
        session = await registry.ensure(None)
        session.busy = True
        with pytest.raises(SessionBusyError):
            await registry.ensure(session.session_id)
        return await registry.ensure(None)

    new_session = asyncio.run(run())

    assert new_session.session_id.startswith("s_")
