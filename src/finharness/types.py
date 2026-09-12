"""Shared data contracts for the FinHarness layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


@dataclass(slots=True)
class ToolUse:
    call_id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolUseDelta:
    index: int
    call_id: str | None = None
    name_delta: str = ""
    arguments_delta: str = ""


@dataclass(slots=True)
class Msg:
    role: str
    content: str | None
    tool_uses: list[ToolUse] = field(default_factory=list)
    tool_results: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def user(cls, content: str) -> "Msg":
        return cls(role="user", content=content)


class StreamEvent(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_USE_DELTA = "tool_use_delta"
    MESSAGE_END = "message_end"
    ERROR = "error"


@dataclass(slots=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_cny: float = 0.0
    tool_uses: list[ToolUse] = field(default_factory=list)


@dataclass(slots=True)
class StreamChunk:
    event: StreamEvent
    data: Any = None


@dataclass(slots=True)
class EngineEvent:
    kind: str
    data: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    content: str
    ok: bool = True
    error: str | None = None
    attachments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentTurnOutcome:
    answer: str
    succeeded: bool = True
    usage: ModelUsage = field(default_factory=ModelUsage)
    error: str | None = None
    reason: str | None = None
    tool_calls: int = 0
    retry_count: int = 0
    tool_duration_ms: int = 0


class OutputSink(Protocol):
    async def emit(self, event: EngineEvent) -> None: ...
