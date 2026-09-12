"""In-memory session lifecycle and single-flight protection.

A *session* is an execution window: it guards against concurrent runs and expires
on TTL. A *conversation* is the memory scope, and it outlives any session (it
lives in the memory store). Keeping them separate is what lets a client resume a
conversation after its session expired — the id the client holds is the
conversation, not the ephemeral session.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


class SessionBusyError(RuntimeError):
    pass


@dataclass
class ServerSession:
    session_id: str
    conversation_id: str
    loop: object
    created_at: float
    last_active: float
    busy: bool = False


def new_conversation_id() -> str:
    return f"c_{uuid.uuid4().hex[:12]}"


class SessionRegistry:
    def __init__(self, loop_factory, ttl_s: int = 1800):
        self.loop_factory = loop_factory
        self.ttl_s = ttl_s
        self.sessions: dict[str, ServerSession] = {}

    async def ensure(
        self, session_id: str | None = None, *, conversation_id: str | None = None
    ) -> ServerSession:
        """Return a session for this request, creating one when needed.

        Resolution order:

        1. a live session matching ``session_id`` (the current execution window);
        2. a live session already bound to ``conversation_id`` (resuming a
           conversation whose previous window is still open);
        3. otherwise a new session, bound to ``conversation_id`` when given or to
           a fresh conversation id.
        """
        now = time.monotonic()
        self._evict_expired(now)

        if session_id:
            session = self.sessions.get(session_id)
            if session and now - session.last_active <= self.ttl_s:
                if session.busy:
                    raise SessionBusyError("session is busy")
                session.last_active = now
                return session

        if conversation_id:
            existing = self._find_by_conversation(conversation_id)
            if existing is not None:
                if existing.busy:
                    raise SessionBusyError("session is busy")
                existing.last_active = now
                return existing

        resolved_conversation = conversation_id or new_conversation_id()
        session_id = f"s_{uuid.uuid4().hex[:12]}"
        session = ServerSession(
            session_id=session_id,
            conversation_id=resolved_conversation,
            loop=self.loop_factory(session_id, resolved_conversation),
            created_at=now,
            last_active=now,
        )
        self.sessions[session_id] = session
        return session

    def _find_by_conversation(self, conversation_id: str) -> ServerSession | None:
        for session in self.sessions.values():
            if session.conversation_id == conversation_id:
                return session
        return None

    def _evict_expired(self, now: float) -> int:
        """Drop sessions past their TTL.

        Without this the registry only ever grew: an expired entry stayed in the
        dict forever, holding its loop and transcript. Conversations persist in
        the memory store, so discarding the in-memory session costs nothing.
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
