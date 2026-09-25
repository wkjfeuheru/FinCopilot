"""Persisted tool batches and crash-recovery classification."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from test_loop import ScriptedProvider, StubRegistry, text_round, tool_round

from finharness.config.settings import ContextSettings, Settings, ToolSettings
from finharness.context.memory.state_store import SqliteAgentStateStore
from finharness.context.memory.store import MemoryStore
from finharness.engine.loop import AgentLoop
from finharness.engine.state import (
    AgentPhase,
    CallStatus,
    PersistedToolCall,
    new_agent_state,
    state_from_dict,
)
from finharness.tools.base import PermissionLevel
from finharness.types import Msg, ToolResult, ToolUse


class RecordingTool:
    """Records executed call_ids (seed args carry call_id for assertion)."""

    permission = PermissionLevel.READ
    timeout = None

    def __init__(self, name: str, *, content: str = "ok") -> None:
        self.name = name
        self.content = content
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult(content=self.content, ok=True)


def call(
    call_id: str,
    *,
    permission: str = "read",
    status: str = "pending",
    result_json: str | None = None,
    name: str = "quote",
) -> PersistedToolCall:
    return PersistedToolCall(
        call_id=call_id,
        name=name,
        args={"call_id": call_id},
        permission=permission,
        status=CallStatus(status),
        result_json=result_json,
    )


def _settings() -> Settings:
    return Settings(
        context=ContextSettings(max_turns=30, max_result_tokens=1000),
        tools=ToolSettings(timeout_default_s=30),
    )


def _memory(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


def load_phase(store: MemoryStore, phase: AgentPhase):
    with store._connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM agent_state_snapshots"
            " WHERE phase = ? ORDER BY id DESC LIMIT 1",
            (phase.value,),
        ).fetchone()
    assert row is not None, f"no snapshot with phase={phase.value}"
    return state_from_dict(json.loads(row["state_json"]))


def build_persistent_tool_loop(tmp_path):
    store = _memory(tmp_path)
    tool = RecordingTool("quote")
    registry = StubRegistry({"quote": tool}, read_only={"quote"})
    provider = ScriptedProvider(
        [
            tool_round(ToolUse("c1", "quote", {"call_id": "c1"})),
            text_round("done"),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        registry=registry,
        settings=_settings(),
        system="test",
        store=store,
        conversation_id="conv",
        user_id="",
    )
    return loop, store


def seed_tool_state(tmp_path, *, calls: tuple[PersistedToolCall, ...]) -> MemoryStore:
    store = _memory(tmp_path)
    store.ensure_conversation("conv", user_id="", title="test")
    store.append_messages("conv", [Msg.user("quote")])
    state = replace(
        new_agent_state(
            run_id="run",
            conversation_id="conv",
            user_id="",
            now="t0",
        ),
        phase=AgentPhase.TOOL_USE,
        revision=1,
        turn=1,
        tool_calls=len(calls),
        calls=calls,
        updated_at="t1",
    )
    SqliteAgentStateStore(store).save(state)
    return store


def _resume_loop(tmp_path, tool: RecordingTool, *, read_only: bool = True) -> AgentLoop:
    store = _memory(tmp_path)
    read_only_names = {tool.name} if read_only else set()
    registry = StubRegistry({tool.name: tool}, read_only=read_only_names)
    provider = ScriptedProvider([text_round("recovered")])
    return AgentLoop(
        provider=provider,
        registry=registry,
        settings=_settings(),
        system="test",
        store=store,
        conversation_id="conv",
        user_id="",
    )


async def resume_loop(tmp_path, tool: RecordingTool):
    loop = _resume_loop(tmp_path, tool, read_only=True)
    return await loop.run("", resume=True)


async def resume_until_confirmation(tmp_path):
    """Drive hydrate + tooluse recovery; stop once phase is awaitingconfirmation."""
    store = _memory(tmp_path)
    tool = RecordingTool("write", content="wrote")
    tool.permission = PermissionLevel.WRITE
    registry = StubRegistry({"write": tool}, read_only=set())
    provider = ScriptedProvider([])
    loop = AgentLoop(
        provider=provider,
        registry=registry,
        settings=_settings(),
        system="test",
        store=store,
        conversation_id="conv",
        user_id="",
    )
    from finharness.engine.machine import AgentStateMachine
    from finharness.engine.state import ResumeRequested
    from finharness.utils.clock import utc_now_iso

    state_store = SqliteAgentStateStore(store)
    snapshot = state_store.latest_resumable("conv", "")
    assert snapshot is not None
    recovered_phase = snapshot.phase
    loop._reset_run_locals()
    loop._last_user_msg = ""
    loop._resume_phase = recovered_phase
    loop._is_resume = True
    machine = AgentStateMachine(snapshot, store=state_store, output=None)
    loop._machine = machine
    await machine.dispatch(ResumeRequested(at=utc_now_iso()))
    hydrate = await loop._effect_hydrate(machine.state)
    await machine.dispatch(hydrate.event, messages=hydrate.messages)
    apply = getattr(loop, "_after_dispatch", None)
    if callable(apply):
        apply(hydrate.messages)
    assert machine.state.phase is AgentPhase.TOOL_USE
    result = await loop._effect_tooluse(machine.state)
    await machine.dispatch(result.event, messages=result.messages)
    if callable(apply):
        apply(result.messages)
    return machine.state


def test_thinking_to_tooluse_atomically_persists_assistant_frame(tmp_path):
    loop, store = build_persistent_tool_loop(tmp_path)
    asyncio.run(loop.run("quote"))
    tool_state = load_phase(store, AgentPhase.TOOL_USE)
    assistant = store.load_messages("conv")[1]
    assert assistant.tool_uses[0].call_id == tool_state.calls[0].call_id


def test_recovery_skips_completed_call_and_retries_pending_read(tmp_path):
    seed_tool_state(
        tmp_path,
        calls=(
            call("done", permission="read", status="completed", result_json='{"ok":true}'),
            call("pending", permission="read", status="running"),
        ),
    )
    tool = RecordingTool("quote")
    outcome = asyncio.run(resume_loop(tmp_path, tool))
    assert [call["call_id"] for call in tool.calls] == ["pending"]
    assert outcome.succeeded is True


def test_recovery_marks_uncommitted_write_as_uncertain(tmp_path):
    seed_tool_state(
        tmp_path,
        calls=(call("write", permission="write", status="running", name="write"),),
    )
    state = asyncio.run(resume_until_confirmation(tmp_path))
    assert state.phase is AgentPhase.AWAITING_CONFIRMATION
    assert state.calls[0].status is CallStatus.UNCERTAIN
    assert "上次执行结果未知" in state.confirmation.prompt


def test_cancel_abort_does_not_clobber_durable_finished_call(tmp_path):
    """Durable ToolCallFinished must win even if _finished_call_ids missed the id."""
    from finharness.engine.machine import AgentStateMachine

    store = _memory(tmp_path)
    store.ensure_conversation("conv", user_id="", title="test")
    durable = '{"ok":true,"content":"kept"}'
    state = replace(
        new_agent_state(
            run_id="run", conversation_id="conv", user_id="", now="t0"
        ),
        phase=AgentPhase.TOOL_USE,
        revision=1,
        turn=1,
        tool_calls=2,
        calls=(
            call("done", status="completed", result_json=durable),
            call("open", status="running"),
        ),
        updated_at="t1",
    )
    machine = AgentStateMachine(state, store=SqliteAgentStateStore(store), output=None)
    loop = AgentLoop(
        provider=ScriptedProvider([]),
        registry=StubRegistry(),
        settings=_settings(),
        system="test",
        store=store,
        conversation_id="conv",
        user_id="",
    )
    loop._machine = machine
    loop._finished_call_ids = set()  # simulate cancel between save and local bookkeeping

    aborted = asyncio.run(loop._persist_aborted_results(machine, state.calls))

    done = next(c for c in machine.state.calls if c.call_id == "done")
    assert done.status is CallStatus.COMPLETED
    assert done.result_json == durable
    by_id = dict(aborted)
    assert by_id["done"] == durable
    assert "cancel" in json.loads(by_id["open"])["error"]


def test_pending_survives_until_model_finished_after_dispatch(tmp_path):
    """User frame must remain flushable if ModelFinished dispatch never runs."""
    from finharness.engine.machine import AgentStateMachine

    store = _memory(tmp_path)
    tool = RecordingTool("quote")
    registry = StubRegistry({"quote": tool}, read_only={"quote"})
    provider = ScriptedProvider(
        [tool_round(ToolUse("c1", "quote", {"call_id": "c1"}))]
    )
    loop = AgentLoop(
        provider=provider,
        registry=registry,
        settings=_settings(),
        system="test",
        store=store,
        conversation_id="conv",
        user_id="",
    )
    store.ensure_conversation("conv", user_id="", title="test")
    initial = new_agent_state(
        run_id="run", conversation_id="conv", user_id="", now="t0"
    )
    machine = AgentStateMachine(initial, store=SqliteAgentStateStore(store), output=None)
    loop._machine = machine

    async def drive():
        await machine.start()
        loop._last_user_msg = "quote"
        hydrate = await loop._effect_hydrate(machine.state)
        await machine.dispatch(hydrate.event, messages=hydrate.messages)
        loop._after_dispatch(hydrate.messages)
        assert any(m.role == "user" for m in loop.memory.pending)
        thinking = machine.state
        assert thinking.phase is AgentPhase.THINKING
        result = await loop._effect_think(thinking)
        # Pending must still hold the user frame until dispatch succeeds.
        assert any(m.role == "user" for m in loop.memory.pending)
        assert any(m.role == "user" for m in result.messages)
        assert any(
            getattr(m, "tool_uses", None) for m in result.messages
        )
        return result

    asyncio.run(drive())
