"""Immutable agent FSM domain state and pure reducer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

from finharness.types import ToolUse

SCHEMA_VERSION = 1


class InvalidTransition(ValueError):
    """Raised when an event is illegal for the current phase."""


class UnsupportedStateVersion(ValueError):
    """Raised when a persisted snapshot schema version is not supported."""


class AgentPhase(str, Enum):
    HYDRATE = "hydrate"
    THINKING = "thinking"
    TOOL_USE = "tooluse"
    AWAITING_CONFIRMATION = "awaitingconfirmation"
    COMPACT = "compact"
    COMPLETE = "complete"
    ERROR = "error"


class CallStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaitingconfirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class PersistedToolCall:
    call_id: str
    name: str
    args: Mapping[str, Any]
    permission: str
    status: CallStatus = CallStatus.PENDING
    result_json: str | None = None


@dataclass(frozen=True, slots=True)
class ConfirmationState:
    prompt: str
    options: tuple[str, ...]
    call_ids: tuple[str, ...]
    kind: str = "permission"
    category: str = ""
    status: str = "pending"
    multi_select: bool = False


@dataclass(frozen=True, slots=True)
class RunOutcome:
    kind: str
    reason: str | None
    resumable: bool


@dataclass(frozen=True, slots=True)
class AgentError:
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class AgentState:
    schema_version: int
    run_id: str
    conversation_id: str
    user_id: str
    revision: int
    phase: AgentPhase
    resume_phase: AgentPhase | None
    turn: int
    input_tokens: int
    output_tokens: int
    retry_count: int
    tool_calls: int
    compactions: int
    calls: tuple[PersistedToolCall, ...]
    confirmation: ConfirmationState | None
    outcome: RunOutcome | None
    error: AgentError | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class UsageDelta:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class HydrationFinished:
    needs_compaction: bool
    resume_phase: AgentPhase | None
    at: str


@dataclass(frozen=True, slots=True)
class CompactionFinished:
    at: str


@dataclass(frozen=True, slots=True)
class ModelFinished:
    answer: str
    tool_uses: tuple[ToolUse, ...]
    usage: UsageDelta
    at: str
    permissions: Mapping[str, str] | None = None
    # 本轮 provider 流在首 chunk 前/整轮作废后重试的次数；累加进
    # ``AgentState.retry_count``，使"每步 state"能反映重试压力。
    retries: int = 0


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    call_id: str
    at: str


@dataclass(frozen=True, slots=True)
class ToolCallFinished:
    call_id: str
    status: CallStatus
    result_json: str | None
    at: str


@dataclass(frozen=True, slots=True)
class ToolBatchFinished:
    at: str


@dataclass(frozen=True, slots=True)
class ConfirmationRequested:
    prompt: str
    options: tuple[str, ...]
    call_ids: tuple[str, ...]
    at: str
    kind: str = "permission"
    category: str = ""
    multi_select: bool = False


@dataclass(frozen=True, slots=True)
class ConfirmationResolved:
    approved: bool
    at: str
    answer: str | None = None


@dataclass(frozen=True, slots=True)
class StopRequested:
    at: str
    reason: str | None = "user_stopped"


@dataclass(frozen=True, slots=True)
class RunCompleted:
    kind: str
    reason: str | None
    resumable: bool
    at: str


@dataclass(frozen=True, slots=True)
class RunFailed:
    kind: str
    message: str
    at: str


@dataclass(frozen=True, slots=True)
class ResumeRequested:
    at: str


AgentEvent = (
    HydrationFinished
    | CompactionFinished
    | ModelFinished
    | ToolCallStarted
    | ToolCallFinished
    | ToolBatchFinished
    | ConfirmationRequested
    | ConfirmationResolved
    | StopRequested
    | RunCompleted
    | RunFailed
    | ResumeRequested
)

_HYDRATE_RESUME_PHASES = frozenset(
    {
        AgentPhase.THINKING,
        AgentPhase.TOOL_USE,
        AgentPhase.AWAITING_CONFIRMATION,
    }
)

_STOP_PHASES = frozenset(
    {
        AgentPhase.THINKING,
        AgentPhase.TOOL_USE,
        AgentPhase.AWAITING_CONFIRMATION,
    }
)

_FAIL_PHASES = frozenset(
    {
        AgentPhase.THINKING,
        AgentPhase.TOOL_USE,
        AgentPhase.AWAITING_CONFIRMATION,
    }
)

_ABANDON_PHASES = frozenset(
    {
        AgentPhase.HYDRATE,
        AgentPhase.COMPACT,
        AgentPhase.THINKING,
        AgentPhase.TOOL_USE,
        AgentPhase.AWAITING_CONFIRMATION,
    }
)


def new_agent_state(
    *,
    run_id: str,
    conversation_id: str,
    user_id: str,
    now: str,
) -> AgentState:
    return AgentState(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        conversation_id=conversation_id,
        user_id=user_id,
        revision=0,
        phase=AgentPhase.HYDRATE,
        resume_phase=None,
        turn=0,
        input_tokens=0,
        output_tokens=0,
        retry_count=0,
        tool_calls=0,
        compactions=0,
        calls=(),
        confirmation=None,
        outcome=None,
        error=None,
        created_at=now,
        updated_at=now,
    )


def _advance(state: AgentState, event: AgentEvent, **changes: Any) -> AgentState:
    return replace(
        state,
        revision=state.revision + 1,
        updated_at=event.at,
        **changes,
    )


def _require(condition: bool, state: AgentState, event: AgentEvent) -> None:
    if not condition:
        raise InvalidTransition(
            f"illegal transition: phase={state.phase.value} event={type(event).__name__}"
        )


def _is_resumable(state: AgentState) -> bool:
    if state.phase is AgentPhase.ERROR:
        return False
    if state.phase is AgentPhase.COMPLETE:
        return state.outcome is not None and state.outcome.resumable
    return True


def _copy_args(args: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(args))


def _persisted_from_tool_use(
    tool: ToolUse, permissions: Mapping[str, str] | None
) -> PersistedToolCall:
    permission = "unknown"
    if permissions is not None and tool.call_id in permissions:
        permission = permissions[tool.call_id]
    return PersistedToolCall(
        call_id=tool.call_id,
        name=tool.name,
        args=_copy_args(tool.args),
        permission=permission,
        status=CallStatus.PENDING,
        result_json=None,
    )


def _replace_call(
    calls: tuple[PersistedToolCall, ...],
    call_id: str,
    **changes: Any,
) -> tuple[PersistedToolCall, ...]:
    updated: list[PersistedToolCall] = []
    found = False
    for call in calls:
        if call.call_id == call_id:
            updated.append(replace(call, **changes))
            found = True
        else:
            updated.append(call)
    if not found:
        raise InvalidTransition(f"unknown call_id: {call_id}")
    return tuple(updated)


def _mark_calls(
    calls: tuple[PersistedToolCall, ...],
    call_ids: tuple[str, ...],
    status: CallStatus,
) -> tuple[PersistedToolCall, ...]:
    wanted = set(call_ids)
    return tuple(
        replace(call, status=status) if call.call_id in wanted else call
        for call in calls
    )


def transition(state: AgentState, event: AgentEvent) -> AgentState:
    if isinstance(event, ResumeRequested):
        _require(_is_resumable(state), state, event)
        return _advance(
            state,
            event,
            phase=AgentPhase.HYDRATE,
            outcome=None,
            error=None,
        )

    if isinstance(event, HydrationFinished):
        _require(state.phase is AgentPhase.HYDRATE, state, event)
        if event.resume_phase is not None:
            _require(event.resume_phase in _HYDRATE_RESUME_PHASES, state, event)
        if event.needs_compaction:
            return _advance(
                state,
                event,
                phase=AgentPhase.COMPACT,
                resume_phase=event.resume_phase,
            )
        if event.resume_phase is not None:
            return _advance(
                state,
                event,
                phase=event.resume_phase,
                resume_phase=None,
            )
        return _advance(
            state,
            event,
            phase=AgentPhase.THINKING,
            resume_phase=None,
        )

    if isinstance(event, CompactionFinished):
        _require(state.phase is AgentPhase.COMPACT, state, event)
        if state.resume_phase in _HYDRATE_RESUME_PHASES:
            phase = state.resume_phase
        else:
            phase = AgentPhase.THINKING
        return _advance(
            state,
            event,
            phase=phase,
            resume_phase=None,
            compactions=state.compactions + 1,
        )

    if isinstance(event, ModelFinished):
        _require(state.phase is AgentPhase.THINKING, state, event)
        usage = event.usage
        input_tokens = state.input_tokens + usage.input_tokens
        output_tokens = state.output_tokens + usage.output_tokens
        retries = state.retry_count + event.retries
        if event.tool_uses:
            calls = tuple(
                _persisted_from_tool_use(tool, event.permissions)
                for tool in event.tool_uses
            )
            return _advance(
                state,
                event,
                phase=AgentPhase.TOOL_USE,
                turn=state.turn + 1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                retry_count=retries,
                tool_calls=state.tool_calls + len(calls),
                calls=calls,
                outcome=None,
                error=None,
            )
        return _advance(
            state,
            event,
            phase=AgentPhase.COMPLETE,
            turn=state.turn + 1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry_count=retries,
            outcome=RunOutcome(kind="succeeded", reason=None, resumable=False),
            error=None,
            confirmation=None,
        )

    if isinstance(event, ToolCallStarted):
        _require(state.phase is AgentPhase.TOOL_USE, state, event)
        return _advance(
            state,
            event,
            calls=_replace_call(state.calls, event.call_id, status=CallStatus.RUNNING),
        )

    if isinstance(event, ToolCallFinished):
        _require(state.phase is AgentPhase.TOOL_USE, state, event)
        if event.status not in {
            CallStatus.COMPLETED,
            CallStatus.FAILED,
            CallStatus.UNCERTAIN,
        }:
            raise InvalidTransition(
                f"illegal ToolCallFinished status: {event.status}"
            )
        return _advance(
            state,
            event,
            calls=_replace_call(
                state.calls,
                event.call_id,
                status=event.status,
                result_json=event.result_json,
            ),
        )

    if isinstance(event, ToolBatchFinished):
        _require(state.phase is AgentPhase.TOOL_USE, state, event)
        return _advance(state, event, phase=AgentPhase.THINKING)

    if isinstance(event, ConfirmationRequested):
        _require(state.phase is AgentPhase.TOOL_USE, state, event)
        confirmation = ConfirmationState(
            prompt=event.prompt,
            options=tuple(event.options),
            call_ids=tuple(event.call_ids),
            kind=event.kind,
            category=event.category,
            status="pending",
            multi_select=event.multi_select,
        )
        wanted = set(event.call_ids)
        calls = tuple(
            call
            if call.call_id not in wanted
            else (
                call
                if call.status is CallStatus.UNCERTAIN
                else replace(call, status=CallStatus.AWAITING_CONFIRMATION)
            )
            for call in state.calls
        )
        return _advance(
            state,
            event,
            phase=AgentPhase.AWAITING_CONFIRMATION,
            confirmation=confirmation,
            calls=calls,
        )

    if isinstance(event, ConfirmationResolved):
        _require(state.phase is AgentPhase.AWAITING_CONFIRMATION, state, event)
        call_ids = () if state.confirmation is None else state.confirmation.call_ids
        if event.approved:
            return _advance(
                state,
                event,
                phase=AgentPhase.TOOL_USE,
                confirmation=None,
                calls=_mark_calls(state.calls, call_ids, CallStatus.PENDING),
            )
        return _advance(
            state,
            event,
            phase=AgentPhase.TOOL_USE,
            confirmation=None,
            calls=_mark_calls(state.calls, call_ids, CallStatus.FAILED),
        )

    if isinstance(event, StopRequested):
        _require(state.phase in _STOP_PHASES, state, event)
        return _advance(
            state,
            event,
            phase=AgentPhase.COMPLETE,
            outcome=RunOutcome(
                kind="stopped",
                reason=event.reason or "user_stopped",
                resumable=True,
            ),
            error=None,
            confirmation=None,
        )

    if isinstance(event, RunCompleted):
        _require(state.phase in _ABANDON_PHASES, state, event)
        return _advance(
            state,
            event,
            phase=AgentPhase.COMPLETE,
            outcome=RunOutcome(
                kind=event.kind, reason=event.reason, resumable=event.resumable
            ),
            error=None,
            confirmation=None,
        )

    if isinstance(event, RunFailed):
        _require(state.phase in _FAIL_PHASES, state, event)
        return _advance(
            state,
            event,
            phase=AgentPhase.ERROR,
            error=AgentError(kind=event.kind, message=event.message),
            outcome=None,
            confirmation=None,
        )

    raise InvalidTransition(f"unknown event type: {type(event)!r}")


def state_to_dict(state: AgentState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": state.schema_version,
        "run_id": state.run_id,
        "conversation_id": state.conversation_id,
        "user_id": state.user_id,
        "revision": state.revision,
        "phase": state.phase.value,
        "resume_phase": None
        if state.resume_phase is None
        else state.resume_phase.value,
        "turn": state.turn,
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "retry_count": state.retry_count,
        "tool_calls": state.tool_calls,
        "compactions": state.compactions,
        "calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "args": _copy_args(call.args),
                "permission": call.permission,
                "status": call.status.value,
                "result_json": call.result_json,
            }
            for call in state.calls
        ],
        "confirmation": None
        if state.confirmation is None
        else {
            "prompt": state.confirmation.prompt,
            "options": list(state.confirmation.options),
            "call_ids": list(state.confirmation.call_ids),
            "kind": state.confirmation.kind,
            "category": state.confirmation.category,
            "status": state.confirmation.status,
            "multi_select": state.confirmation.multi_select,
        },
        "outcome": None if state.outcome is None else asdict(state.outcome),
        "error": None if state.error is None else asdict(state.error),
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }
    return payload


def state_from_dict(payload: Mapping[str, Any]) -> AgentState:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UnsupportedStateVersion(
            f"unsupported schema_version: {version!r}; expected {SCHEMA_VERSION}"
        )
    phase = AgentPhase(payload["phase"])
    resume_raw = payload.get("resume_phase")
    resume_phase = None if resume_raw is None else AgentPhase(resume_raw)
    calls = tuple(
        PersistedToolCall(
            call_id=item["call_id"],
            name=item["name"],
            args=_copy_args(item.get("args") or {}),
            permission=item["permission"],
            status=CallStatus(item["status"]),
            result_json=item.get("result_json"),
        )
        for item in payload.get("calls") or ()
    )
    confirmation_raw = payload.get("confirmation")
    confirmation = None
    if confirmation_raw is not None:
        confirmation = ConfirmationState(
            prompt=confirmation_raw["prompt"],
            options=tuple(confirmation_raw.get("options") or ()),
            call_ids=tuple(confirmation_raw.get("call_ids") or ()),
            kind=confirmation_raw.get("kind", "permission"),
            category=confirmation_raw.get("category", ""),
            status=confirmation_raw.get("status", "pending"),
            multi_select=bool(confirmation_raw.get("multi_select", False)),
        )
    outcome_raw = payload.get("outcome")
    outcome = None if outcome_raw is None else RunOutcome(**outcome_raw)
    error_raw = payload.get("error")
    error = None if error_raw is None else AgentError(**error_raw)
    return AgentState(
        schema_version=SCHEMA_VERSION,
        run_id=payload["run_id"],
        conversation_id=payload["conversation_id"],
        user_id=payload["user_id"],
        revision=int(payload["revision"]),
        phase=phase,
        resume_phase=resume_phase,
        turn=int(payload["turn"]),
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload["output_tokens"]),
        retry_count=int(payload["retry_count"]),
        tool_calls=int(payload["tool_calls"]),
        compactions=int(payload["compactions"]),
        calls=calls,
        confirmation=confirmation,
        outcome=outcome,
        error=error,
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
    )


def public_state_view(state: AgentState) -> dict[str, Any]:
    view: dict[str, Any] = {
        "schema_version": state.schema_version,
        "run_id": state.run_id,
        "conversation_id": state.conversation_id,
        "user_id": state.user_id,
        "revision": state.revision,
        "phase": state.phase.value,
        "turn": state.turn,
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "retry_count": state.retry_count,
        "tool_calls": state.tool_calls,
        "compactions": state.compactions,
        "calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "status": call.status.value,
            }
            for call in state.calls
        ],
        "confirmation": None
        if state.confirmation is None
        else {
            "prompt": state.confirmation.prompt,
            "options": list(state.confirmation.options),
            "call_ids": list(state.confirmation.call_ids),
            "kind": state.confirmation.kind,
            "category": state.confirmation.category,
            "status": state.confirmation.status,
            "multi_select": state.confirmation.multi_select,
        },
        "outcome": None if state.outcome is None else asdict(state.outcome),
        "error": None if state.error is None else asdict(state.error),
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }
    return view
