"""请求级追踪上下文：一次对话的所有日志靠 ``trace_id`` 串联（docs 03.14）。

上下文用 ``ContextVar`` 承载，因此沿 ``asyncio`` 任务传播：服务端在请求入口
生成 id，引擎在 ``create_task`` 出来的子任务里仍能读到它，无需把 id 逐层传递。
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, replace

__all__ = [
    "TraceContext",
    "bind_trace",
    "current_trace",
    "new_trace_id",
    "reset_trace",
    "update_turn",
]


@dataclass(frozen=True, slots=True)
class TraceContext:
    """贯穿一次运行的身份与作用域，供日志与 Span 读取。"""

    trace_id: str = ""
    session_id: str = ""
    conversation_id: str = ""
    turn: int = 0


_TRACE: ContextVar[TraceContext | None] = ContextVar("finharness_trace", default=None)


def new_trace_id() -> str:
    """生成一个新的 trace id，前缀与 ``s_``/``c_`` 的既有风格一致。"""
    return f"tr_{uuid.uuid4().hex[:16]}"


def bind_trace(
    *,
    session_id: str = "",
    conversation_id: str = "",
    trace_id: str | None = None,
    turn: int = 0,
) -> TraceContext:
    """绑定（或覆盖）当前上下文的追踪身份，返回新上下文。

    未提供 ``trace_id`` 时生成一个；已存在时保留原有 id，使同一次运行内多次
    绑定共享同一个 id。
    """

    existing = _TRACE.get()
    resolved = trace_id or (existing.trace_id if existing else "") or new_trace_id()
    context = TraceContext(
        trace_id=resolved,
        session_id=session_id or (existing.session_id if existing else ""),
        conversation_id=conversation_id or (existing.conversation_id if existing else ""),
        turn=turn,
    )
    _TRACE.set(context)
    return context


def current_trace() -> TraceContext | None:
    """读取当前上下文；未绑定请求作用域时返回 None。"""
    return _TRACE.get()


def update_turn(turn: int) -> None:
    """更新当前上下文里的轮次号，使日志能标出"第几轮"。"""
    existing = _TRACE.get()
    if existing is not None:
        _TRACE.set(replace(existing, turn=turn))


def reset_trace(token: object | None = None) -> None:
    """清除当前上下文（主要供测试与后台任务清理使用）。"""
    if token is None:
        _TRACE.set(None)
        return
    _TRACE.reset(token)  # type: ignore[arg-type]
