"""Session-local agent loop: one model turn at a time, read-only tools only."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from finharness.config.settings import Settings
from finharness.engine.cost import SessionStats
from finharness.engine.retry import RetryPolicy, stream_with_retry
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
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.system = system
        self.output = output
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self.stats = stats if stats is not None else SessionStats()
        self.messages: list[Msg] = []
        self.usage = ModelUsage()
        self.turn = 0
        self.active_tool_names: set[str] = set(registry.names())

    async def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.output is not None:
            await self.output.emit(EngineEvent(kind, data))

    def _done_payload(self, *, succeeded: bool, reason: str | None, tool_calls: int) -> dict[str, Any]:
        snapshot = self.stats.snapshot()
        return {
            "succeeded": succeeded,
            "reason": reason,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
            "cost_cny": 0.0,
            "tool_calls": tool_calls,
            "retry_count": snapshot.retry_count,
            "tool_duration_ms": snapshot.tool_duration_ms,
        }

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
                        system=self.system,
                        messages=self.messages,
                        tools=self.registry.schemas(self.active_tool_names),
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
        return json.dumps(
            {
                "ok": bool(result.ok),
                "content": self._truncate(result.content),
                "error": result.error,
            },
            ensure_ascii=False,
        )

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
        if not self.registry.is_read_only(tool_use.name):
            return await self._reject(tool_use, f"tool is not read-only: {tool_use.name}")

        await self._emit(
            "tool_status",
            {"call_id": tool_use.call_id, "name": tool_use.name, "status": "started"},
        )
        timeout = self.settings.tools.timeout_overrides.get(
            tool_use.name, self.settings.tools.timeout_default_s
        )
        started_at = self.stats.now()
        try:
            result = await asyncio.wait_for(tool.run(**tool_use.args), timeout)
        except TimeoutError:
            duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
            return await self._reject(
                tool_use, f"tool timeout after {timeout}s: {tool_use.name}", duration_ms=duration_ms
            )
        except Exception as exc:
            duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
            return await self._reject(tool_use, f"tool failed: {exc}", duration_ms=duration_ms)

        duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
        await self._emit(
            "tool_status",
            {
                "call_id": tool_use.call_id,
                "name": tool_use.name,
                "status": "completed" if result.ok else "failed",
                "ok": bool(result.ok),
                "duration_ms": duration_ms,
            },
        )
        return tool_use.call_id, self._encode(result)
