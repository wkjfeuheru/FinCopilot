"""Phase effect runner: dispatch domain events through the agent FSM."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from finharness.engine.machine import AgentStateMachine
from finharness.engine.state import AgentEvent, AgentPhase, AgentState
from finharness.types import AgentTurnOutcome, Msg

AfterDispatch = Callable[[tuple[Msg, ...]], None]


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
    """Drive phase effects until the machine reaches complete or error."""

    def __init__(
        self,
        machine: AgentStateMachine,
        effects: RunnerEffects,
        *,
        after_dispatch: AfterDispatch | None = None,
    ) -> None:
        self.machine = machine
        self.effects = effects
        self.after_dispatch = after_dispatch
        self._handlers = {
            AgentPhase.HYDRATE: self._handle_hydrate,
            AgentPhase.COMPACT: self._handle_compact,
            AgentPhase.THINKING: self._handle_think,
            AgentPhase.TOOL_USE: self._handle_tools,
            AgentPhase.AWAITING_CONFIRMATION: self._handle_await,
        }

    async def run(self) -> AgentTurnOutcome:
        terminal = {AgentPhase.COMPLETE, AgentPhase.ERROR}
        while self.machine.state.phase not in terminal:
            handler = self._handlers[self.machine.state.phase]
            event, messages = await handler(self.machine.state)
            await self.machine.dispatch(event, messages=messages)
            if self.after_dispatch is not None and messages:
                self.after_dispatch(messages)
        return await self.effects.finish(self.machine.state)

    async def _handle_hydrate(
        self, state: AgentState
    ) -> tuple[AgentEvent, tuple[Msg, ...]]:
        result = await self.effects.hydrate(state)
        return result.event, result.messages

    async def _handle_compact(
        self, state: AgentState
    ) -> tuple[AgentEvent, tuple[Msg, ...]]:
        result = await self.effects.compact(state)
        return result.event, result.messages

    async def _handle_think(
        self, state: AgentState
    ) -> tuple[AgentEvent, tuple[Msg, ...]]:
        result = await self.effects.think(state)
        return result.event, result.messages

    async def _handle_tools(
        self, state: AgentState
    ) -> tuple[AgentEvent, tuple[Msg, ...]]:
        result = await self.effects.use_tools(state)
        return result.event, result.messages

    async def _handle_await(
        self, state: AgentState
    ) -> tuple[AgentEvent, tuple[Msg, ...]]:
        result = await self.effects.await_confirmation(state)
        return result.event, result.messages
