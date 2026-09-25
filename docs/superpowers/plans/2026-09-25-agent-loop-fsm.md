# Agent Loop FSM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the implicit agent loop phases with a persisted immutable FSM driven by domain events, while preserving existing engine, SSE, tool, stop, and evaluation behavior.

**Architecture:** A pure reducer creates a new `AgentState` for every transition. `AgentStateMachine` serializes dispatch, persists each revision before emitting the additive `state` event, and `AgentRunner` executes phase effects. SQLite snapshots support explicit restart recovery, including reissued confirmations and safe handling of uncertain write calls.

**Tech Stack:** Python 3.13, frozen dataclasses, asyncio, SQLite, FastAPI/Pydantic, pytest, React/TypeScript, Vitest.

## Global Constraints

- Use exactly these public phases: `hydrate`, `thinking`, `tooluse`, `awaitingconfirmation`, `compact`, `complete`, `error`.
- Every legal transition returns a new immutable state object with a monotonically increasing revision.
- Persist a transition before scheduling its next side effect or emitting its public `state` event.
- Keep `AgentLoop.run()`, `AgentTurnOutcome`, existing engine events, and SSE event names backward compatible.
- Do not persist provider chunks, text deltas, asyncio primitives, callbacks, sinks, spans, timers, providers, registries, tools, or temporary confirmation request IDs.
- Restart recovery is explicit through `resume=true`; a normal new message abandons an older resumable run.
- Retry unfinished read tools with the original call ID; never silently replay an uncertain write tool.
- A recovered confirmation receives a new request ID.
- User stop is not an error and must remain resumable.
- Preserve assistant tool-use/tool-result pairing and tool-result ordering.
- Do not introduce an external message broker or full event sourcing.
- Do not include unrelated existing workspace changes in task commits.

---

## File Map

**Create**

- `src/finharness/engine/state.py` — immutable FSM data model, domain events, serialization, public view, pure reducer.
- `src/finharness/engine/machine.py` — serialized dispatch, persistence-before-emission, transition errors.
- `src/finharness/engine/runner.py` — phase-effect dispatcher and in-process event loop.
- `src/finharness/context/memory/state_store.py` — persistent and in-memory state-store adapters.
- `tests/engine/test_agent_state.py` — reducer, immutability, serialization, public-view tests.
- `tests/engine/test_agent_machine.py` — dispatch ordering and state-event tests.
- `tests/context/test_agent_state_store.py` — SQLite snapshots, isolation, atomic message commits, retention.
- `tests/engine/test_agent_recovery.py` — restart recovery for model, tools, confirmations, and write uncertainty.
- `frontend/src/lib/agentState.ts` — typed public state view and frontend phase reducer.
- `frontend/src/lib/agentState.test.ts` — frontend phase/replay tests.

**Modify**

- `src/finharness/context/memory/records.py` — snapshot schema constants and persisted record.
- `src/finharness/context/memory/store.py` — schema setup, tenant migration, atomic message/snapshot commit, deletion/export.
- `src/finharness/engine/loop.py` — compatibility facade, effect callbacks, state transitions, recovery.
- `src/finharness/permissions/gate.py` — expose confirmation decisions before awaiting user input.
- `src/finharness/server/api.py` — `resume`, state replay, interaction adapter, history response.
- `src/finharness/eval/runner.py` — connect `RecordingSink` and in-memory FSM execution.
- `src/finharness/eval/capture.py` — eval implementation of the interactive port.
- `frontend/src/api/client.ts` — `resume` request and resumable state typing.
- `frontend/src/components/ChatPanel.tsx` — consume `state` and send explicit resume.
- `frontend/src/components/AgentTrace.tsx` — rebuild phase from persisted state events.
- `frontend/src/components/AgentTrace.test.tsx` — state replay assertions.
- `tests/engine/test_loop.py` — additive state-event sequence coverage.
- `tests/engine/test_checkpoint.py` — snapshot/legacy checkpoint compatibility.
- `tests/engine/test_user_stop.py` — stopped complete state and resume.
- `tests/server/test_interactive.py` — awaiting-confirmation ordering and reissue.
- `tests/server/test_api.py` — `resume` request behavior and history payload.
- `tests/server/test_sse.py` — additive `state` SSE frames.
- `tests/observability/test_trace_store.py` — persisted state event coverage.
- `docs/modules/03.3-engine.md` — public phase and recovery behavior.
- `docs/modules/03.12-server.md` — `resume` request and `state` SSE contract.
- `docs/modules/03.6-context.md` — snapshot persistence and retention.

---

### Task 1: Immutable Domain State and Pure Reducer

**Files:**
- Create: `src/finharness/engine/state.py`
- Create: `tests/engine/test_agent_state.py`

**Interfaces:**
- Produces: `AgentPhase`, `CallStatus`, `PersistedToolCall`, `ConfirmationState`, `RunOutcome`, `AgentError`, `AgentState`, typed domain events, `new_agent_state()`, `transition()`, `state_to_dict()`, `state_from_dict()`, `public_state_view()`.
- Consumes: `ToolUse` values from `finharness.types`, converted to immutable persisted call records.

- [ ] **Step 1: Write reducer tests that fix the public model**

```python
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
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `uv run pytest tests/engine/test_agent_state.py -v`

Expected: FAIL during import because `finharness.engine.state` does not exist.

- [ ] **Step 3: Implement frozen state records and explicit event records**

Implement concrete declarations in `state.py`:

```python
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
class AgentState:
    schema_version: int
    run_id: str
    conversation_id: str
    user_id: str
    revision: int
    phase: AgentPhase
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
```

Define separate frozen event dataclasses for hydration, compaction, model completion, call start/completion, confirmation request/resolution, stop, completion, failure, and resume. Do not use a free-form `dict` event because the reducer must exhaustively validate payloads.

- [ ] **Step 4: Implement the transition table and JSON round-trip**

Use `dataclasses.replace` for every legal branch. Centralize revision/time updates:

```python
def _advance(state: AgentState, event: AgentEvent, **changes: Any) -> AgentState:
    return replace(
        state,
        revision=state.revision + 1,
        updated_at=event.at,
        **changes,
    )
```

`state_to_dict()` must copy mappings to JSON-safe dictionaries. `state_from_dict()` must reject any `schema_version != 1` with `UnsupportedStateVersion`. Map resumed snapshots to `hydrate` only through `ResumeRequested`; do not silently coerce illegal persisted values.

- [ ] **Step 5: Run reducer tests**

Run: `uv run pytest tests/engine/test_agent_state.py -v`

Expected: PASS.

- [ ] **Step 6: Lint the new module**

Run: `uv run ruff check src/finharness/engine/state.py tests/engine/test_agent_state.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/finharness/engine/state.py tests/engine/test_agent_state.py
git commit -m "feat: add immutable agent FSM state"
```

---

### Task 2: SQLite Snapshot Store and Atomic Transition Commits

**Files:**
- Create: `src/finharness/context/memory/state_store.py`
- Create: `tests/context/test_agent_state_store.py`
- Modify: `src/finharness/context/memory/records.py`
- Modify: `src/finharness/context/memory/store.py`

**Interfaces:**
- Consumes: `AgentState`, `state_to_dict()`, `state_from_dict()`.
- Produces: `StateStore` protocol, `MemoryStateStore`, `SqliteAgentStateStore`, `save(state, messages=())`, `latest_resumable(conversation_id, user_id)`, `latest_for_run(run_id, user_id)`, and `abandon_latest(...)`.

- [ ] **Step 1: Write storage, isolation, and transaction tests**

```python
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
    memory, states = stores(tmp_path)
    states.save(sample_state(user_id="u1"))
    assert states.latest_for_run("run", user_id="u2") is None
```

Also add tests proving duplicate `(run_id, revision)` fails, conversation deletion removes snapshots, and `MemoryStateStore` has the same load/save behavior without SQLite.

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/context/test_agent_state_store.py -v`

Expected: FAIL because the schema and adapters do not exist.

- [ ] **Step 3: Add schema constants and record type**

In `records.py`, define:

```python
SCHEMA_AGENT_STATE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS agent_state_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    phase TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    resumable INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, revision),
    FOREIGN KEY (user_id, conversation_id)
      REFERENCES conversations(user_id, conversation_id) ON DELETE CASCADE
)
"""
INDEX_AGENT_STATES_BY_CONVERSATION = (
    "CREATE INDEX IF NOT EXISTS idx_agent_states_conversation "
    "ON agent_state_snapshots(user_id, conversation_id, id DESC)"
)
INDEX_AGENT_STATES_BY_RUN = (
    "CREATE INDEX IF NOT EXISTS idx_agent_states_run "
    "ON agent_state_snapshots(user_id, run_id, revision DESC)"
)
```

Add a frozen `AgentStateSnapshot` record containing the indexed fields and decoded `AgentState`.

- [ ] **Step 4: Add transactional primitives to MemoryStore**

Extract the existing message insertion body into `_append_messages(connection, conversation_id, messages)` so both `append_messages()` and FSM commits use one implementation. Add:

```python
def commit_agent_transition(
    self, state: AgentState, *, messages: tuple[Msg, ...] = ()
) -> None:
    with self._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        self._append_messages(connection, state.conversation_id, messages)
        insert_agent_state(connection, state)
```

Create the new schema and indexes in `MemoryStore.__init__` **after**
`self._migrate(connection)` has normalized the parent `conversations` table, then include
the snapshot table in deletion, user export/purge, and conversation retention paths.
Because the snapshot table is new, it is not renamed/copied by the legacy tenant-child
migration.

- [ ] **Step 5: Implement persistent and in-memory adapters**

`SqliteAgentStateStore` delegates atomic writes to `MemoryStore.commit_agent_transition`. Its reads use bound `user_id` and identity fields. `MemoryStateStore` stores snapshots in a list keyed by `(user_id, conversation_id, run_id)` and rejects duplicate revisions.

`latest_resumable()` must inspect the newest snapshot and return it only when the state is nonterminal or has `outcome.resumable is True`; it must not resurrect an older resumable snapshot hidden behind a newer abandoned/successful terminal snapshot.

- [ ] **Step 6: Run storage and existing memory tests**

Run: `uv run pytest tests/context/test_agent_state_store.py tests/context/test_checkpoint_store.py tests/context/test_conversation_memory.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/finharness/context/memory/state_store.py src/finharness/context/memory/records.py src/finharness/context/memory/store.py tests/context/test_agent_state_store.py
git commit -m "feat: persist agent FSM snapshots"
```

---

### Task 3: Serialized State Machine and Additive `state` Event

**Files:**
- Create: `src/finharness/engine/machine.py`
- Create: `tests/engine/test_agent_machine.py`

**Interfaces:**
- Consumes: `AgentState`, typed `AgentEvent`, `transition()`, `StateStore`, `OutputSink`.
- Produces: `AgentStateMachine.state`, `async start() -> AgentState`, `async dispatch(event, messages=()) -> AgentState`.

- [ ] **Step 1: Write persistence and emission ordering tests**

```python
async def test_dispatch_persists_before_emitting_state_event():
    order: list[str] = []
    store = RecordingStore(order)
    sink = RecordingSink(order)
    machine = AgentStateMachine(sample_hydrate_state(), store=store, output=sink)
    next_state = await machine.dispatch(HydrationFinished(False, None, at="t1"))
    assert order == ["persist:1", "emit:state:1"]
    assert machine.state is next_state


async def test_output_failure_does_not_rollback_committed_state():
    store = RecordingStore([])
    machine = AgentStateMachine(
        sample_hydrate_state(), store=store, output=FailingSink()
    )
    next_state = await machine.dispatch(HydrationFinished(False, None, at="t1"))
    assert store.saved[-1] == next_state
    assert machine.state == next_state
```

Add a concurrent-dispatch test using `asyncio.gather` and assert revisions become `[1, 2]`, never duplicate.
Add a `start()` test asserting revision 0 is persisted and emitted exactly once; a second
`start()` call must return the same state without another write or event.

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/engine/test_agent_machine.py -v`

Expected: FAIL because `AgentStateMachine` does not exist.

- [ ] **Step 3: Implement machine dispatch**

```python
class AgentStateMachine:
    def __init__(self, state, *, store, output=None):
        self.state = state
        self.store = store
        self.output = output
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self):
        async with self._lock:
            if self._started:
                return self.state
            self.store.save(self.state)
            self._started = True
            await self._emit_state(self.state)
            return self.state

    async def dispatch(self, event, *, messages=()):
        async with self._lock:
            next_state = transition(self.state, event)
            self.store.save(next_state, messages=tuple(messages))
            self.state = next_state
            await self._emit_state(next_state)
            return next_state
```

`_emit_state()` sends `EngineEvent("state", public_state_view(state))`, re-raises
`CancelledError`, and logs/swallow other sink errors. The machine owns only serialization
and ordering; it must not execute provider/tool effects.

- [ ] **Step 4: Run machine and reducer tests**

Run: `uv run pytest tests/engine/test_agent_machine.py tests/engine/test_agent_state.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/finharness/engine/machine.py tests/engine/test_agent_machine.py
git commit -m "feat: dispatch persisted agent state transitions"
```

---

### Task 4: Phase Runner and Non-Tool Loop Integration

**Files:**
- Create: `src/finharness/engine/runner.py`
- Modify: `src/finharness/engine/loop.py`
- Modify: `tests/engine/test_loop.py`
- Modify: `tests/context/test_compaction.py`

**Interfaces:**
- Consumes: `AgentStateMachine`, effect callbacks returning typed domain events.
- Produces: `AgentRunner.run()`, with phase handlers for hydrate, compact, thinking, complete, and error.

- [ ] **Step 1: Add state-sequence tests for a text-only run**

```python
def test_text_only_run_emits_explicit_fsm_states():
    outcome, events = asyncio.run(run_text_loop("answer"))
    phases = [e.data["phase"] for e in events if e.kind == "state"]
    assert phases == ["hydrate", "thinking", "complete"]
    assert outcome.answer == "answer"
    assert legacy_kinds(events)[-3:] == ["text_delta", "answer", "done"]


def test_compaction_is_an_explicit_phase_before_thinking():
    events = asyncio.run(run_forced_compaction())
    phases = [e.data["phase"] for e in events if e.kind == "state"]
    assert phases[:3] == ["hydrate", "compact", "thinking"]
```

Preserve existing assertions for deltas, answer, done, provider retries, max turns, and provider error.

- [ ] **Step 2: Run focused tests and confirm the missing state events**

Run: `uv run pytest tests/engine/test_loop.py::test_text_only_run_emits_explicit_fsm_states tests/context/test_compaction.py -v`

Expected: the new phase assertion FAILS; existing compaction tests remain green.

- [ ] **Step 3: Implement the deep runner interface**

```python
@dataclass(frozen=True, slots=True)
class EffectResult:
    event: AgentEvent
    messages: tuple[Msg, ...] = ()


@dataclass(slots=True)
class RunnerEffects:
    hydrate: Callable[[AgentState], Awaitable[EffectResult]]
    compact: Callable[[AgentState], Awaitable[EffectResult]]
    think: Callable[[AgentState], Awaitable[EffectResult]]
    use_tools: Callable[[AgentState], Awaitable[EffectResult]]
    await_confirmation: Callable[[AgentState], Awaitable[EffectResult]]
    finish: Callable[[AgentState], Awaitable[AgentTurnOutcome]]


class AgentRunner:
    async def run(self) -> AgentTurnOutcome:
        while self.machine.state.phase not in {AgentPhase.COMPLETE, AgentPhase.ERROR}:
            handler = self._handlers[self.machine.state.phase]
            event, messages = await handler(self.machine.state)
            await self.machine.dispatch(event, messages=messages)
        return await self.effects.finish(self.machine.state)
```

Use a small `EffectResult(event, messages=())` record rather than returning untyped tuples. Keep temporary deltas, first-token timing, observer spans, and provider chunks local to effect methods.

- [ ] **Step 4: Convert loop entry and non-tool branches**

In `AgentLoop.run`, construct:

- `SqliteAgentStateStore(self.store)` when a `MemoryStore` exists.
- `MemoryStateStore()` otherwise.
- initial hydrate `AgentState`.
- `AgentStateMachine` with the existing output sink.
- `AgentRunner` with callbacks implemented by focused `AgentLoop` methods.

Split conversation-row creation from memory hydration: when SQLite is present,
`run()` must call `store.ensure_conversation(...)` before `machine.start()` so the
revision-0 snapshot satisfies its foreign key. Loading messages, citations, summaries,
semantic recall, checkpoints, and routed methodology remains inside the hydrate effect.

Call `machine.start()` for a new run so revision 0 `hydrate` is durable and visible.
For a recovered run, initialize from the latest snapshot and dispatch
`ResumeRequested` so the new `hydrate` revision is durable before any recovery effect.

Move existing hydrate preparation into `_effect_hydrate`, `_maybe_compact` result handling into `_effect_compact`, and one provider stream into `_effect_think`. Preserve existing `text_delta`, `text_reset`, answer, error, done, trace, usage, observer, and retry calls.

- [ ] **Step 5: Run non-tool loop, compaction, stop, and checkpoint tests**

Run: `uv run pytest tests/engine/test_loop.py tests/context/test_compaction.py tests/engine/test_user_stop.py tests/engine/test_checkpoint.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/finharness/engine/runner.py src/finharness/engine/loop.py tests/engine/test_loop.py tests/context/test_compaction.py
git commit -m "refactor: drive agent loop through explicit phases"
```

---

### Task 5: Persisted Tool Batches and Safe Restart Recovery

**Files:**
- Modify: `src/finharness/engine/loop.py`
- Modify: `src/finharness/engine/runner.py`
- Create: `tests/engine/test_agent_recovery.py`
- Modify: `tests/engine/test_loop.py`
- Modify: `tests/engine/test_checkpoint.py`

**Interfaces:**
- Consumes: `PersistedToolCall`, call events, machine atomic message commits.
- Produces: recoverable tool batches; `_effect_tooluse(state) -> EffectResult`.

- [ ] **Step 1: Write tool persistence and crash-recovery tests**

```python
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
        calls=(call("write", permission="write", status="running"),),
    )
    state = asyncio.run(resume_until_confirmation(tmp_path))
    assert state.phase is AgentPhase.AWAITING_CONFIRMATION
    assert state.calls[0].status is CallStatus.UNCERTAIN
    assert "上次执行结果未知" in state.confirmation.prompt
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/engine/test_agent_recovery.py -v`

Expected: FAIL because tool batches are not restored from snapshots.

- [ ] **Step 3: Persist tool-use frame before execution**

When a model response contains tool calls:

1. Create immutable `PersistedToolCall` entries in model order.
2. Dispatch `ModelFinished(tool_uses=...)`.
3. Atomically include the assistant `Msg(tool_uses=...)`.
4. Only after dispatch succeeds, schedule `_execute_one` tasks.

Do not append the assistant frame to mutable memory before the database transition succeeds.

- [ ] **Step 4: Persist each completion and finalize in source order**

Replace a single opaque `gather` result update with wrapped tasks that publish `ToolCallFinished` events as they complete. After every call is terminal, create one tool-result `Msg` sorted by the original tuple order and dispatch `ToolBatchFinished` with that message atomically.

Cancellation must create persisted aborted results for every call lacking a result before re-raising.

- [ ] **Step 5: Implement recovery classification**

On explicit resume:

- `completed`/`failed`: reuse `result_json`.
- pending/running read: return to pending and execute with original call ID.
- pending/running write: mark `uncertain`, create a confirmation stating that the prior result is unknown, and do not execute.
- unknown permission: treat as write/unsafe and require confirmation.

- [ ] **Step 6: Run tool, guard, trace, governance, and recovery tests**

Run: `uv run pytest tests/engine/test_agent_recovery.py tests/engine/test_loop.py tests/engine/test_loop_guard.py tests/engine/test_loop_trace.py tests/engine/test_governance_chain.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/finharness/engine/loop.py src/finharness/engine/runner.py tests/engine/test_agent_recovery.py tests/engine/test_loop.py tests/engine/test_checkpoint.py
git commit -m "feat: recover persisted tool batches safely"
```

---

### Task 6: Explicit Awaiting-Confirmation Port

**Files:**
- Modify: `src/finharness/permissions/gate.py`
- Modify: `src/finharness/engine/loop.py`
- Modify: `src/finharness/engine/runner.py`
- Modify: `src/finharness/server/api.py`
- Modify: `src/finharness/eval/capture.py`
- Modify: `tests/permissions/test_gate.py`
- Modify: `tests/server/test_interactive.py`
- Modify: `tests/engine/test_agent_recovery.py`

**Interfaces:**
- Produces: `ConfirmationSpec`, `PermissionGate.decide(tool, args)`, `PermissionGate.resolve(spec, answer)`, `InteractivePort.prompt(spec)`.
- Preserves: compatibility `PermissionGate.check()` for direct callers by implementing it through `decide/resolve`.

- [ ] **Step 1: Write ordering and reissue tests**

```python
def test_confirmation_state_precedes_interactive_request(tmp_path):
    events = asyncio.run(run_confirming_tool(tmp_path, answer="y"))
    kinds = [event.kind for event in events]
    awaiting = next(i for i, event in enumerate(events)
                    if event.kind == "state"
                    and event.data["phase"] == "awaitingconfirmation")
    request = kinds.index("interactive_request")
    assert awaiting < request


def test_restart_reissues_confirmation_with_new_request_id(tmp_path):
    first_id = seed_and_capture_pending_confirmation(tmp_path)
    second_id, outcome = asyncio.run(resume_and_confirm(tmp_path, "y"))
    assert second_id != first_id
    assert outcome.succeeded is True
```

Also test `y_remember` and `y_session` still update the correct category sets.

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/server/test_interactive.py tests/engine/test_agent_recovery.py -v`

Expected: new ordering/reissue tests FAIL.

- [ ] **Step 3: Split permission decision from interaction**

Add:

```python
@dataclass(frozen=True, slots=True)
class ConfirmationSpec:
    kind: str
    prompt: str
    options: tuple[str, ...]
    category: str
    call_ids: tuple[str, ...] = ()
    multi_select: bool = False
```

`PermissionGate.decide()` performs deny/cache/mode/session checks and returns `GateDecision(Verdict.CONFIRM, confirmation=spec)` without awaiting. `resolve()` maps answers to allow/deny and updates `confirmed_categories` or `session_approved`. `check()` remains a compatibility adapter for existing direct callers with injected callbacks.

Use these exact signatures:

```python
def decide(self, tool, args: dict) -> GateDecision: ...
def resolve(self, spec: ConfirmationSpec, answer: str | None) -> GateDecision: ...

class InteractivePort(Protocol):
    async def prompt(self, spec: ConfirmationSpec) -> str | None: ...
```

- [ ] **Step 4: Route all prompts through the runner**

For gate confirmations and `AskUserTool`:

1. Dispatch `ConfirmationRequested`.
2. Call `InteractivePort.prompt(spec)`, which uses ConfirmBus or eval policy.
3. Dispatch `ConfirmationResolved` before executing a permitted tool.
4. Preserve `interactive_request` and `interaction_resolved` legacy events.

Do not persist ConfirmBus `request_id`; only the adapter owns it.

- [ ] **Step 5: Run permission and interaction suites**

Run: `uv run pytest tests/permissions/test_gate.py tests/server/test_interactive.py tests/engine/test_agent_recovery.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/finharness/permissions/gate.py src/finharness/engine/loop.py src/finharness/engine/runner.py src/finharness/server/api.py src/finharness/eval/capture.py tests/permissions/test_gate.py tests/server/test_interactive.py tests/engine/test_agent_recovery.py
git commit -m "feat: model user confirmation as an FSM phase"
```

---

### Task 7: Server Resume Contract, History, Replay, and Legacy Checkpoints

**Files:**
- Modify: `src/finharness/server/api.py`
- Modify: `tests/server/test_api.py`
- Modify: `tests/server/test_sse.py`
- Modify: `tests/engine/test_checkpoint.py`
- Modify: `tests/observability/test_trace_store.py`

**Interfaces:**
- Consumes: `StateStore.latest_resumable`, `AgentLoop.run(user_msg, *, resume=False)`.
- Produces: `ChatRequest.resume: bool = False`, history `state`/`resumable` payload, replayed `state` events.

- [ ] **Step 1: Write API behavior tests**

```python
def test_chat_request_defaults_resume_to_false():
    request = ChatRequest(message="new")
    assert request.resume is False


def test_normal_message_abandons_resumable_run(client, seeded_state):
    stream_all(client, {"conversation_id": "c1", "message": "new question"})
    assert latest_state("old_run").outcome.kind == "abandoned"
    assert latest_state_for_conversation("c1").run_id != "old_run"


def test_resume_keeps_run_id_and_increments_revision(client, seeded_state):
    events = stream_all(
        client, {"conversation_id": "c1", "message": "", "resume": True}
    )
    states = [event["data"] for event in events if event["event"] == "state"]
    assert states[0]["run_id"] == "old_run"
    assert states[0]["revision"] > seeded_state.revision
```

Add history assertions for safe public state and `resumable`.

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/server/test_api.py tests/server/test_sse.py tests/engine/test_checkpoint.py -v`

Expected: new resume tests FAIL because `ChatRequest` lacks the field and loop lacks explicit recovery.

- [ ] **Step 3: Add explicit resume routing**

Change the model:

```python
class ChatRequest(BaseModel):
    conversation_id: str | None = None
    session_id: str | None = None
    message: str
    resume: bool = False
```

Pass `resume` to `AgentLoop.run(user_msg: str, *, resume: bool = False)`.
`resume=true` requires a matching resumable snapshot; return HTTP 409 if none exists.
A normal request abandons the latest resumable run before creating a new hydrate state.

- [ ] **Step 4: Extend history and replay**

Return:

```json
{
  "conversation_id": "c1",
  "messages": [],
  "resumable": {
    "reason": "user_stopped",
    "rounds": 2,
    "updated_at": "...",
    "state": {"run_id": "r1", "revision": 7, "phase": "complete"}
  }
}
```

Add `state` to `QueueSink.replay_events`; TraceStore already records every non-delta event, so assert rather than duplicate its logic.

- [ ] **Step 5: Keep legacy checkpoints readable**

If no FSM snapshot exists, continue building the old `resumable` response from `TurnCheckpoint`. On first explicit resume, hydrate the old plan into a new FSM run; once a snapshot exists it is authoritative and the old checkpoint cannot override it.

- [ ] **Step 6: Run server, stop, trace, and checkpoint tests**

Run: `uv run pytest tests/server/test_api.py tests/server/test_sse.py tests/server/test_stop.py tests/engine/test_checkpoint.py tests/observability/test_trace_store.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/finharness/server/api.py tests/server/test_api.py tests/server/test_sse.py tests/engine/test_checkpoint.py tests/observability/test_trace_store.py
git commit -m "feat: expose explicit agent run recovery"
```

---

### Task 8: Frontend State Reducer and Explicit Resume

**Files:**
- Create: `frontend/src/lib/agentState.ts`
- Create: `frontend/src/lib/agentState.test.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/components/ChatPanel.tsx`
- Modify: `frontend/src/components/AgentTrace.tsx`
- Modify: `frontend/src/components/AgentTrace.test.tsx`

**Interfaces:**
- Consumes: public `state` SSE/history payload.
- Produces: `AgentPhase`, `PublicAgentState`, `reduceAgentState()`, and `streamChat(..., resume=false)`.

- [ ] **Step 1: Write frontend reducer and request tests**

```typescript
it("keeps the newest state revision", () => {
  const older = state({ run_id: "r1", revision: 2, phase: "thinking" });
  const newer = state({ run_id: "r1", revision: 3, phase: "tooluse" });
  expect(reduceAgentState(newer, older)).toEqual(older);
  expect(reduceAgentState(older, newer)).toEqual(newer);
});

it("maps stopped complete state without treating it as an error", () => {
  const value = state({
    phase: "complete",
    outcome: { kind: "stopped", reason: "user_stopped", resumable: true },
  });
  expect(presentAgentPhase(value)).toEqual({ status: "stopped", label: "已停止" });
});
```

Add a client test or fetch mock asserting resume requests serialize `{resume:true}`.

- [ ] **Step 2: Run and confirm failure**

Run: `npm test -- --run src/lib/agentState.test.ts src/components/AgentTrace.test.tsx`

Working directory: `frontend`

Expected: FAIL because the reducer module does not exist.

- [ ] **Step 3: Implement typed state consumption**

Define the exact phase union and safe payload:

```typescript
export type AgentPhase =
  | "hydrate"
  | "thinking"
  | "tooluse"
  | "awaitingconfirmation"
  | "compact"
  | "complete"
  | "error";

export type PublicAgentState = {
  run_id: string;
  revision: number;
  phase: AgentPhase;
  turn: number;
  calls: Array<{ call_id: string; name: string; status: string }>;
  outcome?: { kind: string; reason: string | null; resumable: boolean } | null;
  error?: { kind: string; message: string } | null;
};
```

Ignore an event whose run differs from the active run or whose revision is not newer.

- [ ] **Step 4: Send explicit resume and preserve old events**

Change `streamChat` to accept `resume = false` and include it in JSON. `ChatPanel` passes `true` only from the existing “继续研究” action. A typed `state` handler updates the high-level phase/status; existing tool/progress/interaction handlers continue rendering detailed activity.

`traceFromStoredTurn` should consume the last state event for final phase while retaining existing `done` metrics and detailed steps.

- [ ] **Step 5: Run frontend tests and build**

Run: `npm test`

Working directory: `frontend`

Expected: PASS.

Run: `npm run build`

Working directory: `frontend`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/agentState.ts frontend/src/lib/agentState.test.ts frontend/src/api/client.ts frontend/src/components/ChatPanel.tsx frontend/src/components/AgentTrace.tsx frontend/src/components/AgentTrace.test.tsx
git commit -m "feat: render persisted agent phases"
```

---

### Task 9: Eval Wiring, Documentation, and Full Verification

**Files:**
- Modify: `src/finharness/eval/runner.py`
- Modify: `src/finharness/eval/capture.py`
- Modify: `tests/eval/test_eval_harness.py`
- Modify: `docs/modules/03.12-server.md`
- Modify: `docs/modules/03.6-context.md`
- Modify: `docs/modules/03.3-engine.md`

**Interfaces:**
- Consumes: in-memory state store, `RecordingSink`, interaction adapter, final server contract.
- Produces: eval captures containing `state` events and updated public documentation.

- [ ] **Step 1: Write the eval event-capture regression**

```python
def test_eval_records_explicit_agent_states(tmp_path):
    run = asyncio.run(run_offline_case(tmp_path, answer="ok"))
    phases = [
        event.data["phase"]
        for event in run.turns[0].events
        if event.kind == "state"
    ]
    assert phases == ["hydrate", "thinking", "complete"]
```

- [ ] **Step 2: Run and confirm the current sink wiring failure**

Run: `uv run pytest tests/eval/test_eval_harness.py -v`

Expected: the new phase assertion FAILS because `RecordingSink` is constructed but not assigned to `AgentLoop.output`.

- [ ] **Step 3: Wire eval through the same event interface**

Pass `output=sink` when constructing `AgentLoop` in `eval/runner.py`. The existing
eval `MemoryStore` therefore uses `SqliteAgentStateStore`; eval callers that omit
`store` continue to receive `MemoryStateStore` from `AgentLoop`. Make
`InteractionChannel` implement `InteractivePort.prompt`. Reset only recorded events
between turns; do not reset the loop’s state-store history.

- [ ] **Step 4: Update documentation with exact contracts**

Document:

- the seven phases and legal transitions;
- immutable revisioned snapshots;
- `state` SSE payload and ordering;
- explicit `resume` request behavior;
- confirmation reissue with a new request ID;
- read retry versus uncertain write reconfirmation;
- snapshot retention and conversation deletion.

Do not claim exactly-once execution for external side effects.

- [ ] **Step 5: Run focused backend verification**

Run:

```bash
uv run pytest \
  tests/engine \
  tests/context/test_agent_state_store.py \
  tests/context/test_compaction.py \
  tests/server/test_api.py \
  tests/server/test_sse.py \
  tests/server/test_interactive.py \
  tests/server/test_stop.py \
  tests/observability/test_trace_store.py \
  tests/eval/test_eval_harness.py -v
```

Expected: PASS.

- [ ] **Step 6: Run static checks**

Run: `uv run ruff check src tests`

Expected: PASS.

- [ ] **Step 7: Run the complete backend suite**

Run: `uv run pytest`

Expected: PASS with smoke tests deselected by project configuration.

- [ ] **Step 8: Run complete frontend verification**

Run: `npm test`

Working directory: `frontend`

Expected: PASS.

Run: `npm run build`

Working directory: `frontend`

Expected: PASS.

- [ ] **Step 9: Inspect the final diff for unrelated files**

Run: `git status --short`

Expected: only files listed in this plan are part of the FSM implementation; pre-existing unrelated workspace modifications remain uncommitted and are not staged.

- [ ] **Step 10: Commit**

```bash
git add src/finharness/eval/runner.py src/finharness/eval/capture.py tests/eval/test_eval_harness.py docs/modules
git commit -m "docs: document persisted agent FSM"
```

