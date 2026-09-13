"""Session-local agent loop: one model turn at a time, read-only tools only."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from finharness.config.settings import Settings
from finharness.context.compaction import AutoCompactor, CompactionResult
from finharness.context.memory.short_term import Episode, ShortTermMemory
from finharness.context.memory.summary import SummaryLayer
from finharness.context.memory.working import WorkingMemory
from finharness.context.session import ResearchContext
from finharness.data.cache import make_lookup_key
from finharness.data.citation import CitationRegistry, fingerprint_frame, fingerprint_text
from finharness.engine.cost import SessionStats
from finharness.engine.retry import RetryPolicy, stream_with_retry
from finharness.hooks.base import HookChain
from finharness.permissions.gate import ReadOnlyGate
from finharness.permissions.modes import Verdict
from finharness.provider.base import Provider
from finharness.tools.registry import ToolRegistry
from finharness.types import (
    AgentTurnOutcome,
    EngineEvent,
    ModelUsage,
    Msg,
    OutputSink,
    StreamEvent,
    ToolResult,
    ToolUse,
)


class _CompactionMarker:
    """Stand-in tool for audit records that describe a compaction."""

    name = "context_compaction"


@dataclass(slots=True)
class RepeatVerdict:
    """Outcome of a repeat check for one tool call."""

    count: int
    escalate: bool
    message: str


class LoopDetected(RuntimeError):
    """Raised when the same tool call repeats beyond the allowed threshold."""

    def __init__(self, message: str, *, tool: str = "") -> None:
        super().__init__(message)
        self.tool = tool


class AgentLoop:
    """Holds one session's conversation, usage and tool budget."""

    TRUNCATION_MARKER = "\n[truncated]"

    def __init__(
        self,
        *,
        provider: Provider,
        registry: ToolRegistry,
        settings: Settings,
        system: str,
        output: OutputSink | None = None,
        retry_policy: RetryPolicy | None = None,
        stats: SessionStats | None = None,
        cite: CitationRegistry | None = None,
        session_id: str | None = None,
        ctx: ResearchContext | None = None,
        gate: Any | None = None,
        hooks: HookChain | None = None,
        interactive: Any | None = None,
        counter: Any | None = None,
        conversation_id: str | None = None,
        store: Any | None = None,
        coordinator: Any | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.system = system
        self.output = output
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self.stats = stats if stats is not None else SessionStats()
        self.cite = cite if cite is not None else CitationRegistry()
        self.session_id = session_id or "local"
        self.ctx = (
            ctx
            if ctx is not None
            else ResearchContext(cite=self.cite, settings=settings)
        )
        # Defaults keep pre-governance behaviour for callers that inject nothing.
        self.gate = gate if gate is not None else ReadOnlyGate()
        self.hooks = hooks if hooks is not None else HookChain()
        self.interactive = interactive
        # Sub-agent coordinator (docs 03.10). Built on demand from the registry's
        # DataAccess so a caller that injects nothing still gets risk review.
        # Accounting is bound below, once this loop's own counters exist.
        self.coordinator = (
            coordinator if coordinator is not None else self._build_coordinator(counter=counter)
        )
        # L1 memory owns the transcript and its accounting; the loop orchestrates.
        # A counter may be injected so callers can share one vocabulary cache.
        self.memory = WorkingMemory(ctx=self.ctx, settings=settings, counter=counter)
        # Let ctx.append_user/append_tool_result forward here (docs 03.6.2).
        self.ctx.memory = self.memory
        # Conversation memory (docs 03.6.4): the transcript and the memory built
        # on it persist per conversation, so a restart can resume without
        # re-summarising. Absent a store, the loop is memory-less as before.
        self.conversation_id = conversation_id or session_id or "local"
        self.store = store
        self.short_term = ShortTermMemory(cap=settings.context.short_mem_cap)
        self.summary = SummaryLayer.load(
            conversation_id=self.conversation_id,
            store=store,
            counter=self.memory.counter,
            budget_tokens=int(
                settings.context.context_window_tokens
                * settings.context.summary_budget_ratio
            ),
        )
        self.ctx.short_term = self.short_term
        self.ctx.summary = self.summary
        self.ctx.store = store
        self._memory_loaded = False
        self.usage = ModelUsage()
        # Review tokens must appear in the session totals, so the coordinator is
        # wired to the counters now that both exist. A test double may not have
        # the binding hook, which is fine — it simply is not billed.
        binding = getattr(self.coordinator, "bind_accounting", None)
        if binding is not None:
            binding(usage=self.usage, stats=self.stats)
        self.turn = 0
        self.compactions: list[CompactionResult] = []
        # Loop guard state, reset per run: an identical (tool, args) call repeated
        # past the threshold stops adding information, so it is refused.
        self._call_counts: dict[str, int] = {}
        self._reminded: set[str] = set()

    @property
    def messages(self) -> list[Msg]:
        """Read-only view of the transcript (kept for callers and tests)."""
        return self.memory.snapshot()

    def _build_coordinator(self, *, counter: Any | None) -> Any | None:
        """Mint a coordinator from the registry's data access, when one is needed.

        Built only when the catalogue actually contains a tool that asks for it,
        which keeps two cases correct: a caller passing a registry double gets no
        sub-agent machinery, and the reviewer's own narrowed registry — which has
        no ``write_report`` — cannot recursively mint a second reviewer.

        The lazy import avoids an engine→coordinator cycle at module load.
        """
        tools = getattr(self.registry, "tools", None)
        if not tools or not any(
            getattr(tool, "needs_coordinator", False) for tool in tools.values()
        ):
            return None
        data = getattr(self.registry, "data", None)
        if data is None:
            return None
        try:
            from finharness.coordinator import Coordinator
        except Exception:  # noqa: BLE001 - a missing coordinator is not fatal
            return None
        return Coordinator(
            provider=self.provider,
            data=data,
            settings=self.settings,
            cite=self.cite,
            counter=counter,
        )

    async def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.output is not None:
            await self.output.emit(EngineEvent(kind, data))

    def _system_prompt(self) -> str:
        """The static system prompt (docs 03.3).

        Deliberately free of per-turn content: it is byte-identical across every
        request so the provider's prefix cache can retain it along with the
        growing message history. The session's research state used to be
        concatenated here, which moved the cache boundary to the front of the
        conversation and made every turn re-process the whole history.
        """
        return self.system

    def _state_text(self) -> str:
        """The session's research state, rendered fresh for this request (docs 03.6.2)."""
        return self.ctx.state_block()

    def _request_messages(self) -> list[Msg]:
        """History plus the research state as a trailing message.

        The state goes last, not first, for two reasons: it keeps the history a
        stable prefix the cache can reuse, and it puts the most current state
        where the model attends to it most. It is never appended to ``raw`` — it
        is a view of this request, not a transcript entry that persists or gets
        summarised.
        """
        messages = self.memory.snapshot()
        state = self._state_text()
        if state:
            messages.append(Msg(role="user", content=state))
        return messages

    # -- conversation memory (docs 03.6.4) ------------------------------------
    async def _load_memory_if_first_turn(self, user_msg: str) -> None:
        """Load persisted memory once, before the first prompt of this loop.

        Everything loaded here is held on ``ctx`` so subsequent turns render the
        same text — the prompt is measured three times per turn and those
        measurements must agree, which a re-query could break.
        """
        if self._memory_loaded or self.store is None:
            return
        self._memory_loaded = True

        self.store.ensure_conversation(
            self.conversation_id, title=_title_from(user_msg)
        )
        # Transcript, so the model resumes mid-thread rather than from nothing.
        # track=False: replayed history is already stored; buffering it would
        # write it again under duplicate sequence numbers.
        for message in self.store.load_messages(self.conversation_id):
            self.memory.append(message, track=False)
        # Citations must keep their ids: stored summaries name them.
        self.cite.restore(self.store.load_citations(self.conversation_id))
        self.ctx.prior_conclusions = self.store.load_conclusions(self.conversation_id)
        self.ctx.notes = self.store.get_notes()
        for symbol in self.store.load_symbols(self.conversation_id):
            self.ctx.remember_symbol(symbol)
        self.summary = SummaryLayer.load(
            conversation_id=self.conversation_id,
            store=self.store,
            counter=self.memory.counter,
            budget_tokens=int(
                self.settings.context.context_window_tokens
                * self.settings.context.summary_budget_ratio
            ),
        )
        self.ctx.summary = self.summary
        self.ctx.refresh_recall()

    def _persist_turn(self) -> None:
        """Write this turn's messages and memory in one transaction.

        Batching at turn end (rather than per message) keeps writes cheap; a
        crash costs at most the current turn, which is acceptable for process
        memory that is meant to aid, not audit.
        """
        pending = self.memory.pending
        if self.store is None or not pending:
            return
        self.store.append_messages(self.conversation_id, pending)
        self.memory.pending = []
        self.store.save_citations(self.conversation_id, self.cite.all())
        for symbol in self.ctx.symbols:
            self.store.upsert_symbol(self.conversation_id, symbol)
        for conclusion in self.ctx.conclusions:
            self.store.save_conclusion(
                self.conversation_id,
                subject=self._conclusion_subject(conclusion.cids),
                text=conclusion.text,
                cids=conclusion.cids,
            )
        self.store.touch_conversation(self.conversation_id)

    def _conclusion_subject(self, cids: list[str]) -> str:
        """Recall key from the cited data's symbol, so one fact keys stably."""
        for cid in cids:
            citation = self.cite.get(cid)
            if citation is not None and citation.symbol:
                return str(citation.symbol)
        return self.conversation_id

    def _remember_episode(
        self, *, kind: str, subject: str, summary: str, ref: dict | None = None
    ) -> None:
        """Record a structured event for L2 recall."""
        if not subject:
            return
        self.short_term.add(
            Episode(kind=kind, subject=subject, summary=summary, ref=dict(ref or {}))
        )
        # Keep the symbol pool in step with what was actually fetched. Deriving
        # it from citations alone would miss fetches that produced no citation
        # (text payloads), leaving recall with no subjects to search.
        if kind == "data":
            self.ctx.remember_symbol(subject)
        self.ctx.refresh_recall()

    async def _maybe_compact(self) -> None:
        """Fold history when the next request would exceed the window budget."""
        compactor = AutoCompactor(
            provider=self.provider,
            memory=self.memory,
            settings=self.settings,
            system=self._system_prompt(),
            tools=self.registry.schemas(),
            summary=self.summary,
            state_text=self._state_text(),
        )
        if not compactor.needs_compaction():
            return
        result = await compactor.compact()
        if not result.compacted and result.warning is None:
            return
        self.compactions.append(result)
        await self._emit(
            "context_compacted",
            {
                "removed": result.removed,
                "before_tokens": result.before_tokens,
                "after_tokens": result.after_tokens,
                "degraded": result.degraded,
                "warning": result.warning,
                "duration_ms": result.duration_ms,
            },
        )
        # Compaction must appear in the audit trail (docs 4.3, action=compact).
        if self.hooks.hooks:
            try:
                await self.hooks.post(
                    _CompactionMarker(),
                    {},
                    ToolResult(content="", ok=True),
                    action="compact",
                    verdict="allow",
                    duration_ms=result.duration_ms,
                    turn=self.turn,
                )
            except Exception:  # noqa: BLE001 - auditing is best-effort
                pass

    async def _audit_detection(self, detected: LoopDetected) -> None:
        """Record the abort in the audit trail (best-effort, never fatal)."""
        if not self.hooks.hooks:
            return
        try:
            await self.hooks.post(
                _CompactionMarker(),
                {"tool": detected.tool},
                ToolResult(content="", ok=False, error=str(detected)),
                action="loop_detected",
                verdict="abort",
                turn=self.turn,
            )
        except Exception:  # noqa: BLE001 - auditing is best-effort
            pass

    def _window_tokens(self) -> int:
        return self.memory.request_tokens(
            system=self._system_prompt(),
            tools=self.registry.schemas(),
            extra_text=self._state_text(),
        )

    # -- loop guard -----------------------------------------------------------
    def _call_fingerprint(self, tool_use: ToolUse) -> str:
        """Stable identity for a (tool, args) pair; same key means same result.

        Reuses the cache's key builder so argument normalisation (sorting,
        JSON-safe rendering) matches how the data layer already treats calls.
        """
        return make_lookup_key(kind=tool_use.name, params=dict(tool_use.args or {}))

    def _check_repeat(self, tool_use: ToolUse) -> RepeatVerdict | None:
        """Count a call and decide whether to nudge or escalate.

        Counting is cumulative for the whole run: an A/B/A/B pattern is a loop
        too, and a repeated identical call is never informative because its
        result is already available.
        """
        limit = self.settings.context.max_identical_tool_calls
        key = self._call_fingerprint(tool_use)
        self._call_counts[key] = self._call_counts.get(key, 0) + 1
        count = self._call_counts[key]
        if count < limit:
            return None

        complex_task = self.ctx.plan is not None
        if key not in self._reminded:
            # First offence: nudge and let the model correct itself.
            self._reminded.add(key)
            if complex_task:
                message = (
                    f"相同参数的 {tool_use.name} 已调用 {count} 次，结果已在上文。"
                    "若当前方向行不通，请用 research_plan 修订计划后继续，不要重复取数。"
                )
            else:
                message = (
                    f"相同参数的 {tool_use.name} 已调用 {count} 次，结果已在上文（或对应 citation），"
                    "请直接复用该结果作答，不要重复取数。"
                )
            return RepeatVerdict(count=count, escalate=False, message=message)

        # Second offence on the same call shape: stop; the run cannot progress.
        return RepeatVerdict(
            count=count,
            escalate=True,
            message=f"相同参数的 {tool_use.name} 重复调用 {count} 次，已中止本轮。",
        )

    def _partial_answer(self) -> str:
        """Summarize what the run did manage to establish before stopping."""
        citations = self.cite.all()
        conclusions = self.ctx.conclusions
        lines = ["本轮已提前结束（检测到重复调用）。"]
        if conclusions:
            lines.append("已形成的结论：")
            lines.extend(
                f"- {item.text}（依据 {'、'.join(item.cids) or '无'}）"
                for item in conclusions[-5:]
            )
        if citations:
            lines.append(f"已获取 {len(citations)} 份数据，可用 citations 精读。")
        if len(lines) == 1:
            lines.append("尚未形成可复述的结论，未能完成该请求。")
        return "\n".join(lines)

    def _done_payload(
        self, *, succeeded: bool, reason: str | None, tool_calls: int
    ) -> dict[str, Any]:
        snapshot = self.stats.snapshot()
        return {
            "succeeded": succeeded,
            "reason": reason,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                # Prefix-cache split of the input above, when the provider
                # reports it; the optimization's effect is only checkable here.
                "cache_hit_tokens": snapshot.cache_hit_tokens,
                "cache_miss_tokens": snapshot.cache_miss_tokens,
                "cache_hit_ratio": self.stats.cache_hit_ratio(),
            },
            # Distinct from the cumulative usage above: this is how full the
            # window is now, which compaction is supposed to reduce.
            "window_tokens": self._window_tokens(),
            "compactions": len(self.compactions),
            "tool_calls": tool_calls,
            "retry_count": snapshot.retry_count,
            "tool_duration_ms": snapshot.tool_duration_ms,
            # Sub-agent spend, already included in ``usage`` above; this is the
            # breakdown so a client can show where the tokens went (docs 03.10).
            "per_agent": {name: dict(entry) for name, entry in snapshot.per_agent.items()},
            "citations": [item.cid for item in self.cite.all()],
        }

    def _citation_ids(self) -> list[str]:
        return [item.cid for item in self.cite.all()]

    async def _fail(
        self,
        *,
        kind: str,
        message: str,
        reason: str,
        tool_calls: int,
        error: str | None,
        answer: str = "",
    ) -> AgentTurnOutcome:
        """End the run unsuccessfully; ``answer`` may carry partial findings."""
        await self._emit("error", {"kind": kind, "message": message, "reason": reason})
        if answer:
            # The run failed, but it is not empty-handed — surface what it found.
            await self._emit("answer", {"text": answer})
        await self._emit(
            "done",
            self._done_payload(succeeded=False, reason=reason, tool_calls=tool_calls),
        )
        return AgentTurnOutcome(
            answer=answer,
            succeeded=False,
            usage=self.usage,
            error=error,
            reason=reason,
            tool_calls=tool_calls,
            retry_count=self.stats.retry_count,
            tool_duration_ms=self.stats.snapshot().tool_duration_ms,
            citations=self._citation_ids(),
        )

    async def run(self, user_msg: str) -> AgentTurnOutcome:
        # On the conversation's first turn, load persisted memory before the
        # prompt is built. Later turns reuse it (see ctx) rather than re-querying:
        # the system prompt is measured three times per turn and those
        # measurements must agree.
        await self._load_memory_if_first_turn(user_msg)
        # Open this turn's write buffer before appending, so the user message is
        # buffered for persistence rather than discarded by the reset.
        self.memory.pending = []
        if user_msg:
            self.memory.append_user(user_msg)

        # Per-run state: the turn budget is per request, so counters reset with it.
        self.turn = 0
        self._call_counts = {}
        self._reminded = set()

        outcome = await self._run_turns()
        self._persist_turn()
        return outcome

    async def _run_turns(self) -> AgentTurnOutcome:
        tool_calls_total = 0
        for _ in range(self.settings.context.max_turns):
            self.turn += 1
            # Window maintenance happens between turns, never mid-request, and
            # must not stop the conversation if summarising fails.
            await self._maybe_compact()
            deltas: list[str] = []
            tool_uses: list[ToolUse] = []
            try:
                async for chunk in stream_with_retry(
                    lambda: self.provider.stream(
                        system=self._system_prompt(),
                        messages=self._request_messages(),
                        tools=self.registry.schemas(),
                        usage=self.usage,
                    ),
                    policy=self.retry_policy,
                    on_retry=lambda _error, _index, _delay: self.stats.add_retry(),
                ):
                    if chunk.event is StreamEvent.TEXT_DELTA:
                        # Stream each delta as it arrives so the answer appears
                        # while it is generated, not in one burst at the end. The
                        # buffer is still kept to assemble the final answer and the
                        # persisted message.
                        deltas.append(chunk.data)
                        await self._emit("text_delta", {"text": chunk.data})
                    elif chunk.event is StreamEvent.MESSAGE_END and isinstance(
                        chunk.data, ModelUsage
                    ):
                        tool_uses = list(chunk.data.tool_uses)
                        self.usage.input_tokens += chunk.data.input_tokens
                        self.usage.output_tokens += chunk.data.output_tokens
                        self.usage.cache_hit_tokens += chunk.data.cache_hit_tokens
                        self.usage.cache_miss_tokens += chunk.data.cache_miss_tokens
                        self.stats.add_usage(
                            chunk.data.input_tokens,
                            chunk.data.output_tokens,
                            cache_hit_tokens=chunk.data.cache_hit_tokens,
                            cache_miss_tokens=chunk.data.cache_miss_tokens,
                        )
            except Exception as exc:
                return await self._fail(
                    kind=type(exc).__name__,
                    message=str(exc),
                    reason="provider_error",
                    tool_calls=tool_calls_total,
                    error=str(exc),
                )

            if tool_uses:
                # This round turned out to be a tool call, so any text streamed
                # before it was a draft, not the answer. Tell the client to drop it
                # before the tool activity is shown; the final answer streams on a
                # later round.
                if deltas:
                    await self._emit("text_reset", {})
                self.memory.append_assistant(
                    Msg(role="assistant", content=None, tool_uses=tool_uses)
                )
                try:
                    results = list(
                        await asyncio.gather(
                            *(self._execute_one(tool_use) for tool_use in tool_uses)
                        )
                    )
                except LoopDetected as detected:
                    # Pair every call so the transcript stays well-formed, then
                    # end the run with whatever was already established.
                    self.memory.append(
                        Msg(
                            role="tool_result",
                            content=None,
                            tool_results=self._aborted_results(tool_uses),
                        )
                    )
                    await self._audit_detection(detected)
                    return await self._fail(
                        kind="loop_detected",
                        message=str(detected),
                        reason="loop_detected",
                        tool_calls=tool_calls_total,
                        error=str(detected),
                        answer=self._partial_answer(),
                    )
                except BaseException:
                    # An aborted round must not leave the assistant frame without the
                    # paired tool messages, or the next request is malformed.
                    self.memory.append(
                        Msg(
                            role="tool_result",
                            content=None,
                            tool_results=self._aborted_results(tool_uses),
                        )
                    )
                    raise
                self.memory.append(
                    Msg(role="tool_result", content=None, tool_results=results)
                )
                tool_calls_total += len(results)
                continue

            answer = "".join(deltas)
            await self._emit("answer", {"text": answer})
            await self._emit(
                "done",
                self._done_payload(
                    succeeded=True, reason=None, tool_calls=tool_calls_total
                ),
            )
            self.memory.append_assistant(Msg(role="assistant", content=answer))
            return AgentTurnOutcome(
                answer=answer,
                usage=self.usage,
                tool_calls=tool_calls_total,
                retry_count=self.stats.retry_count,
                tool_duration_ms=self.stats.snapshot().tool_duration_ms,
                citations=self._citation_ids(),
            )

        return await self._fail(
            kind="max_turns_exhausted",
            message="maximum agent turns exhausted",
            reason="max_turns_exhausted",
            tool_calls=tool_calls_total,
            error=None,
        )

    def _truncate(self, content: str) -> str:
        """Cap one tool result at ``context.max_result_tokens`` tokens.

        The setting is named in tokens, so the cut is made on a real count rather
        than a character length (a character limit would let Chinese text run
        roughly twice the intended budget).
        """
        limit = self.settings.context.max_result_tokens
        if self.memory.counter.count(content).tokens <= limit:
            return content
        # Binary-search the longest prefix within budget; the count is monotone.
        low, high = 0, len(content)
        while low < high:
            middle = (low + high + 1) // 2
            if self.memory.counter.count(content[:middle]).tokens <= limit:
                low = middle
            else:
                high = middle - 1
        prefix = content[:low]
        if limit <= 4:  # no room for the marker
            return prefix
        return prefix.rstrip() + self.TRUNCATION_MARKER

    def _encode(self, result: ToolResult) -> str:
        payload: dict[str, Any] = {
            "ok": bool(result.ok),
            "content": self._truncate(result.content),
            "error": result.error,
        }
        if result.citations:
            payload["citations"] = list(result.citations)
        return json.dumps(payload, ensure_ascii=False)

    def _aborted_results(self, tool_uses: list[ToolUse]) -> list[tuple[str, str]]:
        """Placeholder failures so a cancelled round still pairs every call id."""

        return [
            (
                tool_use.call_id,
                json.dumps(
                    {
                        "ok": False,
                        "content": "",
                        "error": f"tool call cancelled: {tool_use.name}",
                    },
                    ensure_ascii=False,
                ),
            )
            for tool_use in tool_uses
        ]

    async def _reject(
        self, tool_use: ToolUse, message: str, *, duration_ms: int | None = None
    ) -> tuple[str, str]:
        status: dict[str, Any] = {
            "call_id": tool_use.call_id,
            "name": tool_use.name,
            "status": "failed",
            "ok": False,
            "error": message,
        }
        if duration_ms is not None:
            status["duration_ms"] = duration_ms
        await self._emit("tool_status", status)
        return tool_use.call_id, json.dumps(
            {"ok": False, "content": "", "error": message}, ensure_ascii=False
        )

    async def _execute_one(self, tool_use: ToolUse) -> tuple[str, str]:
        self.stats.record_tool_request(tool_use.name)
        tool = self.registry.resolve(tool_use.name)
        if tool is None:
            return await self._reject(tool_use, f"unknown tool: {tool_use.name}")

        # Loop guard: an identical call past the threshold cannot add information
        # (its result is already in the transcript and in the cache), so it is
        # refused with a nudge instead of being executed again.
        guard = self._check_repeat(tool_use)
        if guard is not None:
            await self._emit(
                "loop_guard",
                {
                    "call_id": tool_use.call_id,
                    "name": tool_use.name,
                    "count": guard.count,
                    "action": "refused" if not guard.escalate else "would_abort",
                },
            )
            if guard.escalate:
                # Second offence for this call shape: stop the run rather than
                # keep paying for a turn that cannot progress.
                raise LoopDetected(
                    f"tool {tool_use.name} repeated with identical arguments "
                    f"{guard.count} times"
                )
            return tool_use.call_id, json.dumps(
                {"ok": False, "content": "", "error": guard.message}, ensure_ascii=False
            )

        # Governance chain (docs 03.3.3): permission verdict, then pre-hooks.
        decision = await self.gate.check(tool, tool_use.args)
        if decision.verdict is Verdict.DENY:
            await self._audit(tool, tool_use.args, action="denied", verdict="deny")
            return await self._reject(
                tool_use, decision.reason or f"tool denied: {tool_use.name}"
            )
        if not await self.hooks.pre(tool, tool_use.args, turn=self.turn):
            await self._audit(tool, tool_use.args, action="denied", verdict="blocked")
            return await self._reject(
                tool_use, f"tool blocked by hook: {tool_use.name}"
            )

        await self._emit(
            "tool_status",
            {"call_id": tool_use.call_id, "name": tool_use.name, "status": "started"},
        )
        # Operator override wins; otherwise the tool's declared budget, falling
        # back to the global default when it declares none (docs 03.3.3).
        default_timeout = self.settings.tools.timeout_default_s
        timeout = self.settings.tools.timeout_overrides.get(
            tool_use.name, tool.timeout or default_timeout
        )
        started_at = self.stats.now()
        if getattr(tool, "needs_interactive", False) and self.interactive is not None:
            tool.interactive = self.interactive
        # Same pattern for tools that spawn a sub-agent (docs 03.10): the tool
        # cannot build one, so the loop hands over the session's coordinator.
        if getattr(tool, "needs_coordinator", False) and self.coordinator is not None:
            tool.coordinator = self.coordinator
        try:
            result = await asyncio.wait_for(tool.run(**tool_use.args), timeout)
        except TimeoutError:
            duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
            await self._audit(
                tool,
                tool_use.args,
                action="run",
                verdict=decision.verdict.value,
                ok=False,
                duration_ms=duration_ms,
            )
            return await self._reject(
                tool_use,
                f"tool timeout after {timeout}s: {tool_use.name}",
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
            await self._audit(
                tool,
                tool_use.args,
                action="run",
                verdict=decision.verdict.value,
                ok=False,
                duration_ms=duration_ms,
            )
            return await self._reject(
                tool_use, f"tool failed: {exc}", duration_ms=duration_ms
            )

        duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
        result.citations = self._register_citations(tool_use, result)
        await self._audit(
            tool,
            tool_use.args,
            action="run",
            verdict=decision.verdict.value,
            ok=bool(result.ok),
            duration_ms=duration_ms,
            citations=result.citations,
            result=result,
        )
        await self._emit(
            "tool_status",
            {
                "call_id": tool_use.call_id,
                "name": tool_use.name,
                "status": "completed" if result.ok else "failed",
                "ok": bool(result.ok),
                "duration_ms": duration_ms,
                "citations": list(result.citations),
                # Produced files (charts, reports) ride along so the client can
                # offer them without a second lookup.
                "attachments": list(result.attachments),
            },
        )
        return tool_use.call_id, self._encode(result)

    async def _audit(
        self,
        tool,
        args: dict,
        *,
        action: str,
        verdict: str,
        ok: bool = True,
        duration_ms: float = 0.0,
        citations: list[str] | None = None,
        result: ToolResult | None = None,
    ) -> None:
        """Emit an audit record; governance failures must never break a turn."""
        if not self.hooks.hooks:
            return
        endpoint, rows, cols = "", 0, 0
        sources = getattr(result, "sources", None) if result is not None else None
        if sources:
            first = sources[0]
            endpoint = getattr(first, "endpoint", "") or ""
            df = getattr(first, "df", None)
            if df is not None:
                rows, cols = int(len(df)), int(len(df.columns))
        try:
            await self.hooks.post(
                tool,
                args,
                result if result is not None else ToolResult(content="", ok=ok),
                action=action,
                verdict=verdict,
                duration_ms=duration_ms,
                citations=list(citations or []),
                turn=self.turn,
                endpoint=endpoint,
                rows=rows,
                cols=cols,
            )
        except Exception:  # noqa: BLE001 - auditing is best-effort
            pass

    def _register_citations(self, tool_use: ToolUse, result: ToolResult) -> list[str]:
        """Turn a tool's raw payloads into session-tracked citation ids."""
        cids: list[str] = []
        symbol = (
            tool_use.args.get("symbol") if isinstance(tool_use.args, dict) else None
        )
        for source in getattr(result, "sources", []) or []:
            df = getattr(source, "df", None)
            citation = self.cite.register(
                tool=tool_use.name,
                endpoint=getattr(source, "endpoint", "") or tool_use.name,
                symbol=str(symbol) if symbol is not None else None,
                params=dict(getattr(source, "params", {}) or {}),
                rows=int(len(df)) if df is not None else 0,
                cols=int(len(df.columns)) if df is not None else 0,
                # Text-only payloads have no frame to digest, so hash the text
                # itself; otherwise every text citation shared one fingerprint.
                fingerprint=(
                    fingerprint_frame(df)
                    if df is not None
                    else fingerprint_text(getattr(source, "text", None))
                ),
                from_cache=bool(getattr(source, "from_cache", False)),
                parquet_path=getattr(source, "parquet_path", None),
            )
            cids.append(citation.cid)
        # L2: record that this data was fetched, with a pointer to it, so a later
        # follow-up can reuse it rather than fetching again. Recorded per call
        # rather than per source, because a fetch happened even when the payload
        # is text (no DataFrame) rather than tabular.
        if symbol and result.ok:
            self._remember_episode(
                kind="data",
                subject=str(symbol),
                summary=f"{tool_use.name} 已取数"
                + (f"（{len(cids)} 份）" if cids else ""),
                ref={"cids": list(cids)},
            )
        return cids


def _title_from(user_msg: str) -> str | None:
    """Derive a conversation title from its opening message."""
    text = (user_msg or "").strip()
    if not text:
        return None
    return text[:40]
