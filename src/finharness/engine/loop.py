"""Minimal model loop shared by HTTP and future CLI entry points."""

from __future__ import annotations

from finharness.provider.base import Provider
from finharness.types import AgentTurnOutcome, EngineEvent, ModelUsage, Msg, StreamEvent


class AgentLoop:
    def __init__(self, *, provider: Provider, output=None, system: str = "You are FinHarness, a careful financial research copilot.", registry=None):
        self.provider = provider
        self.output = output
        self.system = system
        self.messages: list[Msg] = []
        self.usage = ModelUsage()
        self.registry = registry

    async def _emit(self, kind: str, data: dict) -> None:
        if self.output is not None:
            await self.output.emit(EngineEvent(kind, data))

    async def run(self, user_msg: str) -> AgentTurnOutcome:
        if user_msg:
            self.messages.append(Msg.user(user_msg))
        try:
            for _ in range(8):
                answer = ""
                round_tool_uses = []
                async for chunk in self.provider.stream(
                    system=self.system,
                    messages=self.messages,
                    tools=self.registry.schemas() if self.registry else [],
                    usage=self.usage,
                ):
                    if chunk.event is StreamEvent.TEXT_DELTA:
                        answer += chunk.data
                        await self._emit("text_delta", {"text": chunk.data})
                    elif chunk.event is StreamEvent.MESSAGE_END and isinstance(chunk.data, ModelUsage):
                        self.usage.input_tokens += chunk.data.input_tokens
                        self.usage.output_tokens += chunk.data.output_tokens
                        round_tool_uses = chunk.data.tool_uses
                if round_tool_uses and self.registry:
                    self.messages.append(Msg(role="assistant", content=answer, tool_uses=round_tool_uses))
                    tool_results = []
                    for tool_use in round_tool_uses:
                        tool = self.registry.resolve(tool_use.name)
                        await self._emit("tool_status", {"name": tool_use.name, "status": "start"})
                        if tool is None:
                            result = {"ok": False, "error": f"unknown tool: {tool_use.name}"}
                        else:
                            result_obj = await tool.run(**tool_use.args)
                            result = {"ok": result_obj.ok, "content": result_obj.content, "error": result_obj.error}
                        await self._emit("tool_status", {"name": tool_use.name, "status": "done", "ok": result["ok"]})
                        tool_results.append((tool_use.call_id, result.get("content") or result.get("error") or ""))
                    self.messages.append(Msg(role="tool_result", content=None, tool_results=tool_results))
                    continue
                self.messages.append(Msg(role="assistant", content=answer))
                await self._emit("answer", {"text": answer})
                await self._emit("done", {
                    "succeeded": True,
                    "usage": {"input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens},
                    "cost_cny": self.usage.cost_cny,
                })
                return AgentTurnOutcome(answer=answer, usage=self.usage)
            raise RuntimeError("maximum agent turns exhausted")
        except Exception as exc:
            await self._emit("error", {"kind": type(exc).__name__, "message": str(exc)})
            await self._emit("done", {"succeeded": False})
            return AgentTurnOutcome(answer=answer, succeeded=False, usage=self.usage, error=str(exc))
