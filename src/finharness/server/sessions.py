"""In-memory session lifecycle and single-flight protection."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


class SessionBusyError(RuntimeError):
    pass


@dataclass
class ServerSession:
    session_id: str
    loop: object
    created_at: float
    last_active: float
    busy: bool = False


class SessionRegistry:
    def __init__(self, loop_factory, ttl_s: int = 1800):
        self.loop_factory = loop_factory
        self.ttl_s = ttl_s
        self.sessions: dict[str, ServerSession] = {}

    async def ensure(self, session_id: str | None) -> ServerSession:
        now = time.monotonic()
        self._evict_expired(now)
        if session_id:
            session = self.sessions.get(session_id)
            if session and now - session.last_active <= self.ttl_s:
                if session.busy:
                    raise SessionBusyError("session is busy")
                session.last_active = now
                return session
        session_id = f"s_{uuid.uuid4().hex[:12]}"
        session = ServerSession(session_id, self.loop_factory(session_id), now, now)
        self.sessions[session_id] = session
        return session

    def _evict_expired(self, now: float) -> int:
        """Drop sessions past their TTL.

        Without this the registry only ever grew: an expired entry stayed in the
        dict forever, holding its loop and transcript. Conversations now persist
        in the memory store, so discarding the in-memory session costs nothing
        and the entry can be released.
        """
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if not session.busy and now - session.last_active > self.ttl_s
        ]
        for session_id in expired:
            del self.sessions[session_id]
        return len(expired)

    def release(self, session: ServerSession) -> None:
        session.busy = False
        session.last_active = time.monotonic()
