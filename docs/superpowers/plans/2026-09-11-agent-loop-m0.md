# Agent Loop M0 Implementation Plan

**Goal:** Build an offline-verifiable Agent Loop with terminal answers, safe read-only tool calls, structured tool-result feedback, session-local state, and SSE completion/error behavior.

**Global constraints:** Python 3.13 and asyncio only; tests use injected Provider/tool doubles and never call real APIs. Unknown, unregistered, and non-read tools are rejected. A Provider request is one turn and uses `settings.context.max_turns`. Do not implement concrete retry, cost accounting, compaction, confirmation, hooks, audit, or citations. Cancellation propagates `CancelledError` and sends no `done` event.

### Task 1: Contracts and Registry Metadata

**Files:** `src/finharness/types.py`, `src/finharness/tools/registry.py`, `tests/engine/test_loop.py`.

Add `reason: str | None` and `tool_calls: int` to `AgentTurnOutcome`, retaining `error`. Add `ToolRegistry.schemas(names: set[str] | None = None)` and `ToolRegistry.is_read_only(name)`. Explicitly mark only the current three financial tools as read-only. Preserve schema order and existing no-argument behavior. Write failing tests first, observe failure, then implement and run `uv run pytest tests/engine/test_loop.py -q`.

### Task 2: Session-local Agent Loop

**Files:** `src/finharness/engine/loop.py`, `tests/engine/test_loop.py`.

Require injected `provider`, `registry`, `settings`, `system`, with optional `output`. Keep `messages`, accumulated `usage`, turn count, and active tool names private to the Loop/session. Buffer each provider response until `MESSAGE_END`; emit its text only if it contains no tool calls. For tool calls, append an assistant frame with `content=None`, execute each read-only tool through `asyncio.gather`, enforce configured timeouts, emit `tool_status` (`started`, `completed`, `failed`), truncate result content by character limit, and append one JSON-encoded result per call id. Convert unknown, denied, timeout, and tool exceptions into `ok=false` tool results and continue. Return structured reasons for exhausted turns and Provider failures, emitting one error and done event for each; no actual retry or cost computation. Test direct answers, hidden drafts, pairings, concurrency, refusal, exception, timeout, truncation, exhaustion, Provider failure, and cancellation using local doubles.

### Task 3: HTTP/SSE Integration

**Files:** `src/finharness/server/api.py`, `tests/server/test_api.py`.

Inject a stable `DEFAULT_SYSTEM_PROMPT`, Settings, and shared stateless registry into each newly-created session Loop. Preserve session-local history and existing `delta`/`answer` events. Pass through `tool_status`, `error`, and `done`; copy done payload and add session_id at the API edge. On SSE cancellation, cancel and await the Loop task before releasing the session. Add offline endpoint tests for done session ids, hidden tool drafts, tool status, Provider error/done, and multi-turn history.

### Task 4: Verification

Run `git diff --check`, `uv run pytest tests/engine tests/server -q`, then `uv run pytest` (use the project-local `.pytest-tmp` TEMP/TMP workaround if required by the host). Do not alter unrelated dirty files.
