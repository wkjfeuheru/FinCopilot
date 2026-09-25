"""Agent FSM snapshot stores: SQLite-backed and in-memory adapters."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Protocol

from finharness.context.memory.store import MemoryStore, _snapshot_resumable
from finharness.engine.state import (
    AgentPhase,
    AgentState,
    RunCompleted,
    RunOutcome,
    state_from_dict,
    transition,
)
from finharness.types import Msg


class StateStore(Protocol):
    """Persistable agent FSM snapshot surface used by the loop."""

    def save(self, state: AgentState, messages: tuple[Msg, ...] = ()) -> None: ...

    def latest_resumable(
        self, conversation_id: str, user_id: str
    ) -> AgentState | None: ...

    def latest_for_run(self, run_id: str, user_id: str) -> AgentState | None: ...

    def abandon_latest(
        self, conversation_id: str, user_id: str, *, now: str
    ) -> AgentState | None: ...


def _next_abandoned(current: AgentState, now: str) -> AgentState:
    """Next abandoned revision: RunCompleted for nonterminal, replace for user-stop."""
    if current.phase is AgentPhase.COMPLETE:
        return replace(
            current,
            revision=current.revision + 1,
            updated_at=now,
            outcome=RunOutcome(kind="abandoned", reason=None, resumable=False),
            confirmation=None,
        )
    return transition(
        current,
        RunCompleted(kind="abandoned", reason=None, resumable=False, at=now),
    )


class MemoryStateStore:
    """In-memory append-only snapshot store (same API as SqliteAgentStateStore)."""

    def __init__(self) -> None:
        self._snapshots: list[AgentState] = []

    def save(self, state: AgentState, messages: tuple[Msg, ...] = ()) -> None:
        del messages  # in-memory adapter has no message side channel
        for existing in self._snapshots:
            if existing.run_id == state.run_id and existing.revision == state.revision:
                raise ValueError(
                    f"duplicate snapshot revision: run_id={state.run_id!r} "
                    f"revision={state.revision}"
                )
        self._snapshots.append(state)

    def latest_for_run(self, run_id: str, user_id: str) -> AgentState | None:
        latest: AgentState | None = None
        for state in self._snapshots:
            if state.run_id == run_id and state.user_id == user_id:
                if latest is None or state.revision > latest.revision:
                    latest = state
        return latest

    def latest_resumable(
        self, conversation_id: str, user_id: str
    ) -> AgentState | None:
        newest: AgentState | None = None
        newest_index = -1
        for index, state in enumerate(self._snapshots):
            if (
                state.conversation_id == conversation_id
                and state.user_id == user_id
                and index > newest_index
            ):
                newest = state
                newest_index = index
        if newest is None or not _snapshot_resumable(newest):
            return None
        return newest

    def abandon_latest(
        self, conversation_id: str, user_id: str, *, now: str
    ) -> AgentState | None:
        current = self.latest_resumable(conversation_id, user_id)
        if current is None:
            return None
        abandoned = _next_abandoned(current, now)
        self.save(abandoned)
        return abandoned


class SqliteAgentStateStore:
    """SQLite-backed agent FSM snapshots via MemoryStore atomic commits."""

    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory

    def save(self, state: AgentState, messages: tuple[Msg, ...] = ()) -> None:
        self._memory.commit_agent_transition(state, messages=messages)

    def latest_for_run(self, run_id: str, user_id: str) -> AgentState | None:
        with self._memory._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM agent_state_snapshots"
                " WHERE user_id = ? AND run_id = ?"
                " ORDER BY revision DESC LIMIT 1",
                (user_id, run_id),
            ).fetchone()
        if row is None:
            return None
        return state_from_dict(json.loads(row["state_json"]))

    def latest_resumable(
        self, conversation_id: str, user_id: str
    ) -> AgentState | None:
        with self._memory._connect() as connection:
            row = connection.execute(
                "SELECT resumable, state_json FROM agent_state_snapshots"
                " WHERE user_id = ? AND conversation_id = ?"
                " ORDER BY id DESC LIMIT 1",
                (user_id, conversation_id),
            ).fetchone()
        if row is None or not int(row["resumable"]):
            return None
        state = state_from_dict(json.loads(row["state_json"]))
        if not _snapshot_resumable(state):
            return None
        return state

    def abandon_latest(
        self, conversation_id: str, user_id: str, *, now: str
    ) -> AgentState | None:
        current = self.latest_resumable(conversation_id, user_id)
        if current is None:
            return None
        abandoned = _next_abandoned(current, now)
        self.save(abandoned)
        return abandoned
