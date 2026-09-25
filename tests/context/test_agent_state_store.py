"""Agent FSM snapshot store: append-only revisions, atomic commits, isolation."""

from __future__ import annotations

import sqlite3

import pytest

from finharness.context.memory.state_store import MemoryStateStore, SqliteAgentStateStore
from finharness.context.memory.store import MemoryStore
from finharness.engine.state import (
    HydrationFinished,
    ModelFinished,
    RunCompleted,
    StopRequested,
    UsageDelta,
    new_agent_state,
    transition,
)
from finharness.types import Msg, ToolUse


def stores(tmp_path, *, conversation_id: str = "conv", user_id: str = "u"):
    memory = MemoryStore(tmp_path / "memory.db")
    memory.ensure_conversation(conversation_id, user_id=user_id, title="test")
    return memory, SqliteAgentStateStore(memory)


def sample_state(
    *,
    user_id: str = "u",
    run_id: str = "run",
    conversation_id: str = "conv",
    now: str = "t0",
):
    return new_agent_state(
        run_id=run_id,
        conversation_id=conversation_id,
        user_id=user_id,
        now=now,
    )


def thinking_to_tooluse_state():
    hydrate = new_agent_state(run_id="run", conversation_id="conv", user_id="u", now="t0")
    thinking = transition(hydrate, HydrationFinished(False, None, at="t1"))
    return transition(
        thinking,
        ModelFinished(
            answer="",
            tool_uses=(ToolUse(call_id="c1", name="quote", args={}),),
            usage=UsageDelta(),
            at="t2",
        ),
    )


def test_store_appends_revisions_and_loads_latest(tmp_path):
    memory = MemoryStore(tmp_path / "memory.db")
    memory.ensure_conversation("c1", user_id="u1", title="test")
    states = SqliteAgentStateStore(memory)
    first = new_agent_state(run_id="r1", conversation_id="c1", user_id="u1", now="t0")
    second = transition(first, HydrationFinished(False, None, at="t1"))
    states.save(first)
    states.save(second)
    assert states.latest_for_run("r1", user_id="u1") == second


def test_atomic_transition_commits_messages_and_snapshot(tmp_path):
    memory, states = stores(tmp_path)
    state = thinking_to_tooluse_state()
    message = Msg(role="assistant", content=None, tool_uses=[ToolUse("c1", "quote", {})])
    states.save(state, messages=(message,))
    assert memory.load_messages("conv")[-1].tool_uses[0].call_id == "c1"
    assert states.latest_for_run("run", user_id="u") == state


def test_wrong_user_cannot_load_snapshot(tmp_path):
    memory, states = stores(tmp_path, user_id="u1")
    states.save(sample_state(user_id="u1"))
    assert states.latest_for_run("run", user_id="u2") is None


def test_duplicate_run_revision_is_rejected(tmp_path):
    memory, states = stores(tmp_path)
    state = sample_state()
    states.save(state)
    with pytest.raises(sqlite3.IntegrityError):
        states.save(state)


def test_delete_conversation_removes_snapshots(tmp_path):
    memory, states = stores(tmp_path)
    states.save(sample_state())
    assert states.latest_for_run("run", user_id="u") is not None
    memory.delete_conversation("conv")
    assert states.latest_for_run("run", user_id="u") is None


def test_memory_state_store_rejects_duplicates_and_loads_latest():
    states = MemoryStateStore()
    first = sample_state()
    second = transition(first, HydrationFinished(False, None, at="t1"))
    states.save(first)
    states.save(second)
    assert states.latest_for_run("run", user_id="u") == second
    with pytest.raises(ValueError):
        states.save(first)


def test_latest_resumable_ignores_older_when_newest_is_terminal(tmp_path):
    memory, states = stores(tmp_path)
    thinking = transition(sample_state(), HydrationFinished(False, None, at="t1"))
    states.save(thinking)
    assert states.latest_resumable("conv", "u") == thinking

    succeeded = transition(
        thinking,
        ModelFinished(answer="done", tool_uses=(), usage=UsageDelta(), at="t2"),
    )
    states.save(succeeded)
    assert states.latest_resumable("conv", "u") is None


def test_abandon_latest_nonterminal_persists_abandoned_revision(tmp_path):
    memory, states = stores(tmp_path)
    thinking = transition(sample_state(), HydrationFinished(False, None, at="t1"))
    states.save(thinking)
    abandoned = states.abandon_latest("conv", "u", now="t9")
    assert abandoned is not None
    assert abandoned.phase.value == "complete"
    assert abandoned.outcome is not None
    assert abandoned.outcome.kind == "abandoned"
    assert abandoned.outcome.resumable is False
    assert abandoned.revision == thinking.revision + 1
    assert states.latest_resumable("conv", "u") is None
    assert states.latest_for_run("run", user_id="u") == abandoned


def test_abandon_latest_stopped_complete_without_illegal_event(tmp_path):
    memory, states = stores(tmp_path)
    thinking = transition(sample_state(), HydrationFinished(False, None, at="t1"))
    stopped = transition(thinking, StopRequested(at="t2"))
    states.save(stopped)
    assert states.latest_resumable("conv", "u") == stopped

    abandoned = states.abandon_latest("conv", "u", now="t9")
    assert abandoned is not None
    assert abandoned.phase.value == "complete"
    assert abandoned.outcome is not None
    assert abandoned.outcome.kind == "abandoned"
    assert abandoned.outcome.resumable is False
    assert abandoned.revision == stopped.revision + 1
    assert states.latest_resumable("conv", "u") is None


def test_export_and_purge_include_agent_state_snapshots(tmp_path):
    memory, states = stores(tmp_path, user_id="u1")
    states.save(sample_state(user_id="u1"))
    exported = memory.export_user_data(user_id="u1")
    assert "agent_state_snapshots" in exported
    assert len(exported["agent_state_snapshots"]) == 1

    counts = memory.purge_user_data(user_id="u1")
    assert counts["agent_state_snapshots"] == 1
    assert states.latest_for_run("run", user_id="u1") is None


def test_abandon_via_run_completed_also_hides_resumable(tmp_path):
    """Abandoned terminal (reducer path) hides older resumable rows."""
    memory, states = stores(tmp_path)
    thinking = transition(sample_state(), HydrationFinished(False, None, at="t1"))
    states.save(thinking)
    abandoned = transition(
        thinking,
        RunCompleted(kind="abandoned", reason=None, resumable=False, at="t2"),
    )
    states.save(abandoned)
    assert states.latest_resumable("conv", "u") is None
