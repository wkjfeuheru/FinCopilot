"""Serialized agent state machine: persist-then-emit ordering tests."""

from __future__ import annotations

import asyncio

import pytest

from finharness.engine.machine import AgentStateMachine
from finharness.engine.state import (
    HydrationFinished,
    ModelFinished,
    UsageDelta,
    new_agent_state,
)
from finharness.types import EngineEvent


def sample_hydrate_state():
    return new_agent_state(
        run_id="run_1",
        conversation_id="c1",
        user_id="u1",
        now="2026-09-25T09:00:00Z",
    )


class RecordingStore:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.saved: list = []

    def save(self, state, messages=()) -> None:
        self.saved.append(state)
        self.order.append(f"persist:{state.revision}")


class RecordingSink:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.events: list[EngineEvent] = []

    async def emit(self, event: EngineEvent) -> None:
        self.events.append(event)
        self.order.append(f"emit:state:{event.data['revision']}")


class FailingSink:
    async def emit(self, event: EngineEvent) -> None:
        raise RuntimeError("sink failed")


@pytest.mark.asyncio
async def test_dispatch_persists_before_emitting_state_event():
    order: list[str] = []
    store = RecordingStore(order)
    sink = RecordingSink(order)
    machine = AgentStateMachine(sample_hydrate_state(), store=store, output=sink)
    next_state = await machine.dispatch(HydrationFinished(False, None, at="t1"))
    assert order == ["persist:1", "emit:state:1"]
    assert machine.state is next_state


@pytest.mark.asyncio
async def test_output_failure_does_not_rollback_committed_state():
    store = RecordingStore([])
    machine = AgentStateMachine(
        sample_hydrate_state(), store=store, output=FailingSink()
    )
    next_state = await machine.dispatch(HydrationFinished(False, None, at="t1"))
    assert store.saved[-1] == next_state
    assert machine.state == next_state


@pytest.mark.asyncio
async def test_concurrent_dispatch_produces_distinct_successive_revisions():
    store = RecordingStore([])
    machine = AgentStateMachine(sample_hydrate_state(), store=store, output=None)
    results = await asyncio.gather(
        machine.dispatch(HydrationFinished(False, None, at="t1")),
        machine.dispatch(
            ModelFinished(
                answer="done",
                tool_uses=(),
                usage=UsageDelta(),
                at="t2",
            )
        ),
    )
    revisions = [state.revision for state in results]
    assert sorted(revisions) == [1, 2]
    assert len(set(revisions)) == 2
    assert [s.revision for s in store.saved] == sorted(revisions)


@pytest.mark.asyncio
async def test_start_persists_and_emits_revision_zero_exactly_once():
    order: list[str] = []
    store = RecordingStore(order)
    sink = RecordingSink(order)
    machine = AgentStateMachine(sample_hydrate_state(), store=store, output=sink)

    first = await machine.start()
    assert first.revision == 0
    assert order == ["persist:0", "emit:state:0"]
    assert machine.state is first
    assert len(store.saved) == 1
    assert len(sink.events) == 1

    order.clear()
    second = await machine.start()
    assert second is first
    assert order == []
    assert len(store.saved) == 1
    assert len(sink.events) == 1
