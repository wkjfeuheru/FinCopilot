"""交互式请求总线（文档 03.7.1）。

SSE 是单向的，但写工具的确认与 ``ask_user`` 提问都需要模型在轮次中途
暂停，直到用户应答。该总线桥接两者：loop 发出一个 ``interactive_request``
事件并 await 一个 future；客户端通过 ``POST /v1/chat/respond`` 应答，
从而 resolve 它。在 TTL 内无人应答的请求会以超时收场，
使该轮次仍能结束。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


class InteractionTimeout(RuntimeError):
    """请求过期时在内部抛出；调用方拿到的是普通结果。"""


@dataclass(slots=True)
class PendingRequest:
    request_id: str
    session_id: str
    user_id: str
    kind: str  # "confirm" | "question"
    prompt: str
    options: list[str] = field(default_factory=list)
    future: asyncio.Future = field(default=None)  # type: ignore[assignment]


class ConfirmBus:
    """会话作用域的请求注册表，桥接 loop 与 HTTP 客户端。"""

    def __init__(self, *, ttl_s: float = 120.0) -> None:
        self.ttl_s = ttl_s
        self._pending: dict[str, PendingRequest] = {}

    async def request(
        self,
        *,
        session_id: str,
        kind: str,
        prompt: str,
        options: list[str] | None = None,
        ttl_s: float | None = None,
        user_id: str = "",
        announce: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[dict, str | None]:
        """向用户提问；返回 ``(request_payload, answer)``。

        ``announce`` 在请求登记之后、等待应答之前被 await，
        这样传输层可以投递提示而不与客户端的回复竞争。``answer`` 在超时时
        为 ``None``，调用方会将其呈现为显式过期，使模型知道用户从未回复。
        """
        request_id = f"req_{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        pending = PendingRequest(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            kind=kind,
            prompt=prompt,
            options=list(options or []),
            future=loop.create_future(),
        )
        self._pending[request_id] = pending
        payload = {
            "request_id": request_id,
            "kind": kind,
            "prompt": prompt,
            "options": pending.options,
        }
        if announce is not None:
            # 先登记再宣告：快速客户端可能立即应答。
            await announce(payload)
        timeout = ttl_s if ttl_s is not None else self.ttl_s
        try:
            answer = await asyncio.wait_for(pending.future, timeout)
        except (TimeoutError, asyncio.TimeoutError):
            self._pending.pop(request_id, None)
            return payload, None
        self._pending.pop(request_id, None)
        return payload, str(answer)

    def respond(self, *, request_id: str, value: str, user_id: str = "") -> bool:
        """resolve 一个待处理请求；当请求未知、已落定或不属于该用户时返回 False。"""
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return False
        # 交互请求的应答只可能来自挂起它的用户：他人的 request_id
        # 不允许替答写确认或提问（docs 03.13 隔离不变量）。
        if user_id and pending.user_id and pending.user_id != user_id:
            return False
        pending.future.set_result(value)
        return True

    def cancel_session(self, session_id: str) -> int:
        """让某会话的所有待处理请求失败（流中止时使用）。"""
        cancelled = 0
        for request_id, pending in list(self._pending.items()):
            if pending.session_id != session_id:
                continue
            if not pending.future.done():
                pending.future.cancel()
            self._pending.pop(request_id, None)
            cancelled += 1
        return cancelled

    def pending_ids(self) -> list[str]:
        return list(self._pending)
