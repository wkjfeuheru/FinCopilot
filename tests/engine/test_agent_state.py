"""Immutable agent FSM state and pure reducer tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from finharness.engine.state import (
    SCHEMA_VERSION,
    AgentError,
    AgentPhase,
    CallStatus,
    CompactionFinished,
    ConfirmationRequested,
    ConfirmationResolved,
    ConfirmationState,
    HydrationFinished,
    InvalidTransition,
    ModelFinished,
    PersistedToolCall,
    ResumeRequested,
    RunCompleted,
    RunFailed,
    RunOutcome,
    StopRequested,
    ToolBatchFinished,
    ToolCallFinished,
    ToolCallStarted,
    UnsupportedStateVersion,
    UsageDelta,
    new_agent_state,
    public_state_view,
    state_from_dict,
    state_to_dict,
    transition,
)
from finharness.types import ToolUse


def state_with_call(
    *,
    args: dict,
    result_json: str,
    call_id: str = "call_1",
    name: str = "web_search",
    status: CallStatus = CallStatus.COMPLETED,
):
    base = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    return replace(
        base,
        phase=AgentPhase.TOOL_USE,
        calls=(
            PersistedToolCall(
                call_id=call_id,
                name=name,
                args=args,
                permission="read",
                status=status,
                result_json=result_json,
            ),
        ),
    )


def test_transition_returns_a_new_state_and_increments_revision():
    state = new_agent_state(
        run_id="run_1", conversation_id="c1", user_id="u1", now="2026-09-25T09:00:00Z"
    )
    next_state = transition(
        state,
        HydrationFinished(needs_compaction=False, resume_phase=None, at="2026-09-25T09:00:01Z"),
    )
    assert state.phase is AgentPhase.HYDRATE
    assert state.revision == 0
    assert next_state is not state
    assert next_state.phase is AgentPhase.THINKING
    assert next_state.revision == 1


def test_illegal_transition_is_rejected():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.COMPLETE,
        outcome=RunOutcome(kind="succeeded", reason=None, resumable=False),
    )
    with pytest.raises(InvalidTransition):
        transition(state, ModelFinished(answer="late", tool_uses=(), usage=UsageDelta(), at="t1"))


def test_public_view_does_not_expose_tool_arguments_or_result_payload():
    state = state_with_call(
        args={"query": "secret portfolio", "token": "private"},
        result_json='{"content":"private result"}',
    )
    rendered = json.dumps(public_state_view(state), ensure_ascii=False)
    assert "secret portfolio" not in rendered
    assert "private result" not in rendered
    assert public_state_view(state)["calls"][0] == {
        "call_id": "call_1", "name": "web_search", "status": "completed"
    }


def test_new_agent_state_defaults():
    state = new_agent_state(run_id="r1", conversation_id="c1", user_id="u1", now="t0")
    assert state.schema_version == SCHEMA_VERSION == 1
    assert state.revision == 0
    assert state.phase is AgentPhase.HYDRATE
    assert state.resume_phase is None
    assert state.turn == 0
    assert state.calls == ()
    assert state.confirmation is None
    assert state.outcome is None
    assert state.error is None
    assert state.created_at == "t0"
    assert state.updated_at == "t0"


def test_hydrate_to_compact_when_compaction_needed():
    state = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    next_state = transition(
        state, HydrationFinished(needs_compaction=True, resume_phase=None, at="t1")
    )
    assert next_state.phase is AgentPhase.COMPACT
    assert next_state.resume_phase is None
    assert next_state.revision == 1
    assert next_state.updated_at == "t1"


def test_hydrate_compact_then_resume_to_tooluse():
    state = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    compacting = transition(
        state,
        HydrationFinished(
            needs_compaction=True,
            resume_phase=AgentPhase.TOOL_USE,
            at="t1",
        ),
    )
    assert compacting.phase is AgentPhase.COMPACT
    assert compacting.resume_phase is AgentPhase.TOOL_USE
    resumed = transition(compacting, CompactionFinished(at="t2"))
    assert resumed.phase is AgentPhase.TOOL_USE
    assert resumed.resume_phase is None
    assert resumed.compactions == 1


def test_hydrate_compact_then_resume_to_awaiting_confirmation():
    state = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    compacting = transition(
        state,
        HydrationFinished(
            needs_compaction=True,
            resume_phase=AgentPhase.AWAITING_CONFIRMATION,
            at="t1",
        ),
    )
    assert compacting.phase is AgentPhase.COMPACT
    assert compacting.resume_phase is AgentPhase.AWAITING_CONFIRMATION
    resumed = transition(compacting, CompactionFinished(at="t2"))
    assert resumed.phase is AgentPhase.AWAITING_CONFIRMATION
    assert resumed.resume_phase is None


def test_hydrate_rejects_illegal_resume_phase():
    state = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    with pytest.raises(InvalidTransition):
        transition(
            state,
            HydrationFinished(
                needs_compaction=True,
                resume_phase=AgentPhase.COMPLETE,
                at="t1",
            ),
        )


def test_hydrate_to_resume_phase_tooluse():
    state = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    next_state = transition(
        state,
        HydrationFinished(
            needs_compaction=False, resume_phase=AgentPhase.TOOL_USE, at="t1"
        ),
    )
    assert next_state.phase is AgentPhase.TOOL_USE
    assert next_state.resume_phase is None


def test_hydrate_to_awaiting_confirmation_via_resume_phase():
    state = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    next_state = transition(
        state,
        HydrationFinished(
            needs_compaction=False,
            resume_phase=AgentPhase.AWAITING_CONFIRMATION,
            at="t1",
        ),
    )
    assert next_state.phase is AgentPhase.AWAITING_CONFIRMATION
    assert next_state.resume_phase is None


def test_compact_to_thinking_increments_compactions():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.COMPACT,
        revision=1,
        compactions=0,
        resume_phase=None,
    )
    next_state = transition(state, CompactionFinished(at="t2"))
    assert next_state.phase is AgentPhase.THINKING
    assert next_state.resume_phase is None
    assert next_state.compactions == 1
    assert next_state.revision == 2


def test_thinking_to_tooluse_converts_tool_use_records():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.THINKING,
        revision=1,
        turn=0,
    )
    tool = ToolUse(call_id="c1", name="quote", args={"symbol": "AAPL"})
    next_state = transition(
        state,
        ModelFinished(
            answer="",
            tool_uses=(tool,),
            usage=UsageDelta(input_tokens=10, output_tokens=3),
            at="t2",
            permissions={"c1": "read"},
        ),
    )
    assert next_state.phase is AgentPhase.TOOL_USE
    assert next_state.turn == 1
    assert next_state.input_tokens == 10
    assert next_state.output_tokens == 3
    assert next_state.tool_calls == 1
    assert len(next_state.calls) == 1
    call = next_state.calls[0]
    assert isinstance(call, PersistedToolCall)
    assert call.call_id == "c1"
    assert call.name == "quote"
    assert dict(call.args) == {"symbol": "AAPL"}
    assert call.permission == "read"
    assert call.status is CallStatus.PENDING
    assert not isinstance(call, ToolUse)


def test_thinking_to_complete_on_text_answer():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.THINKING,
        revision=1,
    )
    next_state = transition(
        state,
        ModelFinished(
            answer="done",
            tool_uses=(),
            usage=UsageDelta(input_tokens=2, output_tokens=4),
            at="t2",
        ),
    )
    assert next_state.phase is AgentPhase.COMPLETE
    assert next_state.outcome == RunOutcome(
        kind="succeeded", reason=None, resumable=False
    )
    assert next_state.input_tokens == 2
    assert next_state.output_tokens == 4


def test_thinking_to_error_on_run_failed():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.THINKING,
        revision=1,
    )
    next_state = transition(
        state, RunFailed(kind="provider", message="upstream down", at="t2")
    )
    assert next_state.phase is AgentPhase.ERROR
    assert next_state.error == AgentError(kind="provider", message="upstream down")
    assert next_state.outcome is None


def test_tool_call_started_and_finished_update_call_without_phase_error():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.TOOL_USE,
        revision=2,
        calls=(
            PersistedToolCall(
                call_id="c1",
                name="quote",
                args={"symbol": "X"},
                permission="read",
            ),
        ),
    )
    running = transition(state, ToolCallStarted(call_id="c1", at="t3"))
    assert running.phase is AgentPhase.TOOL_USE
    assert running.calls[0].status is CallStatus.RUNNING
    assert running.revision == 3

    failed = transition(
        running,
        ToolCallFinished(
            call_id="c1",
            status=CallStatus.FAILED,
            result_json='{"error":"timeout"}',
            at="t4",
        ),
    )
    assert failed.phase is AgentPhase.TOOL_USE
    assert failed.calls[0].status is CallStatus.FAILED
    assert failed.calls[0].result_json == '{"error":"timeout"}'
    assert failed.error is None


def test_tool_batch_finished_returns_to_thinking():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.TOOL_USE,
        revision=3,
        calls=(
            PersistedToolCall(
                call_id="c1",
                name="quote",
                args={},
                permission="read",
                status=CallStatus.COMPLETED,
                result_json="{}",
            ),
        ),
    )
    next_state = transition(state, ToolBatchFinished(at="t5"))
    assert next_state.phase is AgentPhase.THINKING
    assert next_state.revision == 4


def test_confirmation_request_and_resolution():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.TOOL_USE,
        revision=2,
        calls=(
            PersistedToolCall(
                call_id="w1",
                name="write_file",
                args={"path": "/tmp/x"},
                permission="write",
                status=CallStatus.PENDING,
            ),
        ),
    )
    awaiting = transition(
        state,
        ConfirmationRequested(
            prompt="允许写入？",
            options=("y", "n"),
            call_ids=("w1",),
            kind="permission",
            category="write",
            at="t3",
            multi_select=False,
        ),
    )
    assert awaiting.phase is AgentPhase.AWAITING_CONFIRMATION
    assert awaiting.confirmation == ConfirmationState(
        prompt="允许写入？",
        options=("y", "n"),
        call_ids=("w1",),
        kind="permission",
        category="write",
        status="pending",
        multi_select=False,
    )
    assert awaiting.calls[0].status is CallStatus.AWAITING_CONFIRMATION

    resumed = transition(
        awaiting, ConfirmationResolved(approved=True, answer="y", at="t4")
    )
    assert resumed.phase is AgentPhase.TOOL_USE
    assert resumed.confirmation is None
    assert resumed.calls[0].status is CallStatus.PENDING


def test_confirmation_multi_select_round_trips_through_state():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.TOOL_USE,
        revision=2,
        calls=(
            PersistedToolCall(
                call_id="q1",
                name="ask_user",
                args={"question": "时段？", "multi_select": True},
                permission="read",
                status=CallStatus.PENDING,
            ),
        ),
    )
    awaiting = transition(
        state,
        ConfirmationRequested(
            prompt="时段？",
            options=("近一年", "近三年"),
            call_ids=("q1",),
            kind="question",
            category="question",
            at="t3",
            multi_select=True,
        ),
    )
    assert awaiting.confirmation is not None
    assert awaiting.confirmation.multi_select is True
    restored = state_from_dict(state_to_dict(awaiting))
    assert restored.confirmation is not None
    assert restored.confirmation.multi_select is True
    assert public_state_view(awaiting)["confirmation"]["multi_select"] is True


def test_confirmation_denied_returns_to_tooluse_with_failed_call():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.AWAITING_CONFIRMATION,
        revision=3,
        confirmation=ConfirmationState(
            prompt="允许？",
            options=("y", "n"),
            call_ids=("w1",),
            status="pending",
        ),
        calls=(
            PersistedToolCall(
                call_id="w1",
                name="write_file",
                args={},
                permission="write",
                status=CallStatus.AWAITING_CONFIRMATION,
            ),
        ),
    )
    next_state = transition(
        state, ConfirmationResolved(approved=False, answer="n", at="t4")
    )
    assert next_state.phase is AgentPhase.TOOL_USE
    assert next_state.outcome is None
    assert next_state.confirmation is None
    assert next_state.calls[0].status is CallStatus.FAILED


def test_stop_from_awaiting_confirmation_is_resumable_complete():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.AWAITING_CONFIRMATION,
        revision=3,
        confirmation=ConfirmationState(
            prompt="允许？",
            options=("y", "n"),
            call_ids=("w1",),
            status="pending",
        ),
    )
    next_state = transition(state, StopRequested(at="t4"))
    assert next_state.phase is AgentPhase.COMPLETE
    assert next_state.outcome == RunOutcome(
        kind="stopped", reason="user_stopped", resumable=True
    )


def test_stop_maps_to_complete_stopped_resumable():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.THINKING,
        revision=1,
    )
    next_state = transition(state, StopRequested(at="t2"))
    assert next_state.phase is AgentPhase.COMPLETE
    assert next_state.outcome == RunOutcome(
        kind="stopped", reason="user_stopped", resumable=True
    )
    assert next_state.error is None


def test_abandon_nonterminal_run():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.TOOL_USE,
        revision=2,
    )
    next_state = transition(
        state,
        RunCompleted(kind="abandoned", reason="superseded", resumable=False, at="t3"),
    )
    assert next_state.phase is AgentPhase.COMPLETE
    assert next_state.outcome == RunOutcome(
        kind="abandoned", reason="superseded", resumable=False
    )


def test_resume_requested_from_stopped_complete():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.COMPLETE,
        revision=4,
        outcome=RunOutcome(kind="stopped", reason="user_stopped", resumable=True),
    )
    next_state = transition(state, ResumeRequested(at="t5"))
    assert next_state.phase is AgentPhase.HYDRATE
    assert next_state.revision == 5
    assert next_state.outcome is None


def test_resume_requested_from_nonterminal():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.TOOL_USE,
        revision=3,
    )
    next_state = transition(state, ResumeRequested(at="t4"))
    assert next_state.phase is AgentPhase.HYDRATE


def test_resume_rejected_on_non_resumable_complete():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.COMPLETE,
        outcome=RunOutcome(kind="succeeded", reason=None, resumable=False),
    )
    with pytest.raises(InvalidTransition):
        transition(state, ResumeRequested(at="t1"))


def test_resume_rejected_on_error():
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.ERROR,
        error=AgentError(kind="fatal", message="boom"),
    )
    with pytest.raises(InvalidTransition):
        transition(state, ResumeRequested(at="t1"))


def test_illegal_phase_edges_are_rejected():
    hydrate = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    with pytest.raises(InvalidTransition):
        transition(hydrate, CompactionFinished(at="t1"))

    thinking = replace(hydrate, phase=AgentPhase.THINKING, revision=1)
    with pytest.raises(InvalidTransition):
        transition(thinking, ToolBatchFinished(at="t2"))

    complete = replace(
        thinking,
        phase=AgentPhase.COMPLETE,
        outcome=RunOutcome(kind="succeeded", reason=None, resumable=False),
        revision=2,
    )
    with pytest.raises(InvalidTransition):
        transition(complete, StopRequested(at="t3"))


def test_state_is_frozen():
    state = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    with pytest.raises(Exception):
        state.phase = AgentPhase.THINKING  # type: ignore[misc]


def test_json_round_trip_preserves_state():
    state = replace(
        new_agent_state(run_id="r1", conversation_id="c1", user_id="u1", now="t0"),
        phase=AgentPhase.COMPACT,
        resume_phase=AgentPhase.TOOL_USE,
        revision=2,
        turn=1,
        input_tokens=5,
        output_tokens=7,
        tool_calls=1,
        calls=(
            PersistedToolCall(
                call_id="c1",
                name="quote",
                args={"symbol": "AAPL", "nested": {"a": 1}},
                permission="read",
                status=CallStatus.COMPLETED,
                result_json='{"ok":true}',
            ),
        ),
        confirmation=ConfirmationState(
            prompt="ok?",
            options=("y", "n"),
            call_ids=("c1",),
            kind="permission",
            category="read",
            status="pending",
        ),
        updated_at="t2",
    )
    payload = state_to_dict(state)
    assert isinstance(payload["calls"][0]["args"], dict)
    assert payload["calls"][0]["args"] is not state.calls[0].args
    assert payload["resume_phase"] == "tooluse"
    restored = state_from_dict(payload)
    assert restored == state
    assert restored is not state
    assert restored.resume_phase is AgentPhase.TOOL_USE


def test_persisted_call_args_are_deep_copied():
    nested = {"symbol": "AAPL", "filters": {"market": "US"}}
    original = ToolUse(call_id="c1", name="quote", args=nested)
    state = replace(
        new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0"),
        phase=AgentPhase.THINKING,
        revision=1,
    )
    next_state = transition(
        state,
        ModelFinished(
            answer="",
            tool_uses=(original,),
            usage=UsageDelta(),
            at="t2",
            permissions={"c1": "read"},
        ),
    )
    call_args = next_state.calls[0].args
    assert call_args == nested
    assert call_args is not nested
    assert call_args["filters"] is not nested["filters"]
    nested["filters"]["market"] = "mutated"
    assert call_args["filters"]["market"] == "US"
    payload = state_to_dict(next_state)
    payload["calls"][0]["args"]["filters"]["market"] = "payload-mutated"
    assert next_state.calls[0].args["filters"]["market"] == "US"


def test_state_from_dict_rejects_unsupported_schema_version():
    state = new_agent_state(run_id="r", conversation_id="c", user_id="u", now="t0")
    payload = state_to_dict(state)
    payload["schema_version"] = 99
    with pytest.raises(UnsupportedStateVersion):
        state_from_dict(payload)


def test_public_view_shape_omits_sensitive_fields():
    state = replace(
        state_with_call(
            args={"query": "secret"},
            result_json='{"content":"hidden"}',
        ),
        outcome=RunOutcome(kind="stopped", reason="user_stopped", resumable=True),
        confirmation=ConfirmationState(
            prompt="confirm?",
            options=("y",),
            call_ids=("call_1",),
        ),
    )
    view = public_state_view(state)
    assert set(view["calls"][0]) == {"call_id", "name", "status"}
    assert "args" not in view
    assert "resume_phase" not in view
    assert "result_json" not in json.dumps(view)
    assert view["phase"] == "tooluse"
    assert view["outcome"] == {
        "kind": "stopped",
        "reason": "user_stopped",
        "resumable": True,
    }
    assert view["confirmation"] == {
        "prompt": "confirm?",
        "options": ["y"],
        "call_ids": ["call_1"],
        "kind": "permission",
        "category": "",
        "status": "pending",
        "multi_select": False,
    }
