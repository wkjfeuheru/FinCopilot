"""Interactive request bus (docs 03.7.1).

SSE is one-directional, but both a write-tool confirmation and an ``ask_user``
question need the model to pause mid-turn until the user answers. The bus
bridges them: the loop emits an ``interactive_request`` event and awaits a
future; the client answers through ``POST /v1/chat/respond``, which resolves it.
A request that nobody answers within the TTL resolves as a timeout so the turn
can still finish.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


class InteractionTimeout(RuntimeError):
    """Raised internally when a request expires; callers get a plain result."""


@dataclass(slots=True)
class PendingRequest:
    request_id: str
    session_id: str
    kind: str  # "confirm" | "question"
    prompt: str
    options: list[str] = field(default_factory=list)
    future: asyncio.Future = field(default=None)  # type: ignore[assignment]


class ConfirmBus:
    """Session-scoped request registry bridging the loop and the HTTP client."""

    def __init__(self, *, ttl_s: float = 120.0) -> None:
        self.ttl_s = ttl_s
        self._pending: dict[str, PendingRequest] = {}

    async def request(
        self,
        *,
        session_id: str,
        kind: str,
        prompt: str,
        options: list[str] | None = None,
        ttl_s: float | None = None,
        announce: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[dict, str | None]:
        """Ask the user something; returns ``(request_payload, answer)``.

        ``announce`` is awaited after the request is registered but before the
        answer is awaited, so a transport can deliver the prompt without racing
        the client's reply. ``answer`` is ``None`` on timeout, which the caller
        renders as an explicit expiry so the model knows the user never replied.
        """
        request_id = f"req_{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        pending = PendingRequest(
            request_id=request_id,
            session_id=session_id,
            kind=kind,
            prompt=prompt,
            options=list(options or []),
            future=loop.create_future(),
        )
        self._pending[request_id] = pending
        payload = {
            "request_id": request_id,
            "kind": kind,
            "prompt": prompt,
            "options": pending.options,
        }
        if announce is not None:
            # Registered before announcing: a fast client may answer immediately.
            await announce(payload)
        timeout = ttl_s if ttl_s is not None else self.ttl_s
        try:
            answer = await asyncio.wait_for(pending.future, timeout)
        except (TimeoutError, asyncio.TimeoutError):
            self._pending.pop(request_id, None)
            return payload, None
        self._pending.pop(request_id, None)
        return payload, str(answer)

    def respond(self, *, request_id: str, value: str) -> bool:
        """Resolve a pending request; returns False when it is unknown or settled."""
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return False
        pending.future.set_result(value)
        return True

    def cancel_session(self, session_id: str) -> int:
        """Fail every pending request for a session (used when a stream aborts)."""
        cancelled = 0
        for request_id, pending in list(self._pending.items()):
            if pending.session_id != session_id:
                continue
            if not pending.future.done():
                pending.future.cancel()
            self._pending.pop(request_id, None)
            cancelled += 1
        return cancelled

    def pending_ids(self) -> list[str]:
        return list(self._pending)
