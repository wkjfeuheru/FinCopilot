"""Serialized agent FSM: persist each revision, then emit public state."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from finharness.context.memory.state_store import StateStore
from finharness.engine.state import AgentEvent, AgentState, public_state_view, transition
from finharness.observability.logs import get_logger
from finharness.types import EngineEvent, Msg, OutputSink

log = get_logger("finharness.engine.machine")


class AgentStateMachine:
    """Serialize dispatch, persist before emit; no provider/tool side effects."""

    def __init__(
        self,
        state: AgentState,
        *,
        store: StateStore,
        output: OutputSink | None = None,
    ) -> None:
        self.state = state
        self.store = store
        self.output = output
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> AgentState:
        async with self._lock:
            if self._started:
                return self.state
            self.store.save(self.state)
            self._started = True
            await self._emit_state(self.state)
            return self.state

    async def dispatch(
        self,
        event: AgentEvent,
        *,
        messages: Sequence[Msg] = (),
    ) -> AgentState:
        async with self._lock:
            next_state = transition(self.state, event)
            self.store.save(next_state, messages=tuple(messages))
            self.state = next_state
            await self._emit_state(next_state)
            return next_state

    async def _emit_state(self, state: AgentState) -> None:
        if self.output is None:
            return
        try:
            await self.output.emit(EngineEvent("state", public_state_view(state)))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - sink failure must not roll back committed state
            log.warning(
                "state_event_emit_failed",
                extra={"run_id": state.run_id, "revision": state.revision},
            )
