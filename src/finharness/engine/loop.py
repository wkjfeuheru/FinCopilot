"""Session-local agent loop: one model turn at a time, read-only tools only."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.citation import CitationRegistry, fingerprint_frame
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
        self.ctx = ctx if ctx is not None else ResearchContext(cite=self.cite, settings=settings)
        # Defaults keep pre-governance behaviour for callers that inject nothing.
        self.gate = gate if gate is not None else ReadOnlyGate()
        self.hooks = hooks if hooks is not None else HookChain()
        self.interactive = interactive
        self.messages: list[Msg] = []
        self.usage = ModelUsage()
        self.turn = 0

    async def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.output is not None:
            await self.output.emit(EngineEvent(kind, data))

    def _system_prompt(self) -> str:
        """Base prompt plus the session's research state (docs 03.6.2)."""
        return self.system + self.ctx.state_block()

    def _done_payload(self, *, succeeded: bool, reason: str | None, tool_calls: int) -> dict[str, Any]:
        snapshot = self.stats.snapshot()
        return {
            "succeeded": succeeded,
            "reason": reason,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
            "tool_calls": tool_calls,
            "retry_count": snapshot.retry_count,
            "tool_duration_ms": snapshot.tool_duration_ms,
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
    ) -> AgentTurnOutcome:
        await self._emit("error", {"kind": kind, "message": message, "reason": reason})
        await self._emit("done", self._done_payload(succeeded=False, reason=reason, tool_calls=tool_calls))
        return AgentTurnOutcome(
            answer="",
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
        if user_msg:
            self.messages.append(Msg.user(user_msg))

        tool_calls_total = 0
        for _ in range(self.settings.context.max_turns):
            self.turn += 1
            deltas: list[str] = []
            tool_uses: list[ToolUse] = []
            try:
                async for chunk in stream_with_retry(
                    lambda: self.provider.stream(
                        system=self._system_prompt(),
                        messages=self.messages,
                        tools=self.registry.schemas(),
                        usage=self.usage,
                    ),
                    policy=self.retry_policy,
                    on_retry=lambda _error, _index, _delay: self.stats.add_retry(),
                ):
                    if chunk.event is StreamEvent.TEXT_DELTA:
                        deltas.append(chunk.data)
                    elif chunk.event is StreamEvent.MESSAGE_END and isinstance(chunk.data, ModelUsage):
                        tool_uses = list(chunk.data.tool_uses)
                        self.usage.input_tokens += chunk.data.input_tokens
                        self.usage.output_tokens += chunk.data.output_tokens
                        self.stats.add_usage(chunk.data.input_tokens, chunk.data.output_tokens)
            except Exception as exc:
                return await self._fail(
                    kind=type(exc).__name__,
                    message=str(exc),
                    reason="provider_error",
                    tool_calls=tool_calls_total,
                    error=str(exc),
                )

            if tool_uses:
                self.messages.append(Msg(role="assistant", content=None, tool_uses=tool_uses))
                try:
                    results = list(
                        await asyncio.gather(*(self._execute_one(tool_use) for tool_use in tool_uses))
                    )
                except BaseException:
                    # An aborted round must not leave the assistant frame without the
                    # paired tool messages, or the next request is malformed.
                    self.messages.append(
                        Msg(
                            role="tool_result",
                            content=None,
                            tool_results=self._aborted_results(tool_uses),
                        )
                    )
                    raise
                self.messages.append(Msg(role="tool_result", content=None, tool_results=results))
                tool_calls_total += len(results)
                continue

            answer = "".join(deltas)
            for delta in deltas:
                await self._emit("text_delta", {"text": delta})
            await self._emit("answer", {"text": answer})
            await self._emit(
                "done", self._done_payload(succeeded=True, reason=None, tool_calls=tool_calls_total)
            )
            self.messages.append(Msg(role="assistant", content=answer))
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
        limit = self.settings.context.max_result_tokens
        if len(content) <= limit:
            return content
        if limit <= len(self.TRUNCATION_MARKER):
            return content[:limit]
        return content[: limit - len(self.TRUNCATION_MARKER)] + self.TRUNCATION_MARKER

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

        # Governance chain (docs 03.3.3): permission verdict, then pre-hooks.
        decision = await self.gate.check(tool, tool_use.args)
        if decision.verdict is Verdict.DENY:
            await self._audit(tool, tool_use.args, action="denied", verdict="deny")
            return await self._reject(tool_use, decision.reason or f"tool denied: {tool_use.name}")
        if not await self.hooks.pre(tool, tool_use.args, turn=self.turn):
            await self._audit(tool, tool_use.args, action="denied", verdict="blocked")
            return await self._reject(tool_use, f"tool blocked by hook: {tool_use.name}")

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
        try:
            result = await asyncio.wait_for(tool.run(**tool_use.args), timeout)
        except TimeoutError:
            duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
            await self._audit(
                tool, tool_use.args, action="run", verdict=decision.verdict.value,
                ok=False, duration_ms=duration_ms,
            )
            return await self._reject(
                tool_use, f"tool timeout after {timeout}s: {tool_use.name}", duration_ms=duration_ms
            )
        except Exception as exc:
            duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
            await self._audit(
                tool, tool_use.args, action="run", verdict=decision.verdict.value,
                ok=False, duration_ms=duration_ms,
            )
            return await self._reject(tool_use, f"tool failed: {exc}", duration_ms=duration_ms)

        duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
        result.citations = self._register_citations(tool_use, result)
        await self._audit(
            tool, tool_use.args, action="run", verdict=decision.verdict.value,
            ok=bool(result.ok), duration_ms=duration_ms, citations=result.citations,
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
                tool, args, result if result is not None else ToolResult(content="", ok=ok),
                action=action, verdict=verdict, duration_ms=duration_ms,
                citations=list(citations or []), turn=self.turn,
                endpoint=endpoint, rows=rows, cols=cols,
            )
        except Exception:  # noqa: BLE001 - auditing is best-effort
            pass

    def _register_citations(self, tool_use: ToolUse, result: ToolResult) -> list[str]:
        """Turn a tool's raw payloads into session-tracked citation ids."""
        cids: list[str] = []
        symbol = tool_use.args.get("symbol") if isinstance(tool_use.args, dict) else None
        for source in getattr(result, "sources", []) or []:
            df = getattr(source, "df", None)
            citation = self.cite.register(
                tool=tool_use.name,
                endpoint=getattr(source, "endpoint", "") or tool_use.name,
                symbol=str(symbol) if symbol is not None else None,
                params=dict(getattr(source, "params", {}) or {}),
                rows=int(len(df)) if df is not None else 0,
                cols=int(len(df.columns)) if df is not None else 0,
                fingerprint=fingerprint_frame(df),
                from_cache=bool(getattr(source, "from_cache", False)),
                parquet_path=getattr(source, "parquet_path", None),
            )
            cids.append(citation.cid)
        return cids
