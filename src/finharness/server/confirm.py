"""交互式请求总线（文档 03.7.1）。

SSE 是单向的，但写工具的确认与 ``ask_user`` 提问都需要模型在轮次中途
暂停，直到用户应答。该总线桥接两者：loop 发出一个 ``interactive_request``
事件并 await 一个 future；客户端通过 ``POST /v1/chat/respond`` 应答，
从而 resolve 它。在 TTL 内无人应答的请求会以超时收场，
使该轮次仍能结束。

同一轮里的多个工具调用是并发派发的（``asyncio.gather``），因此三个
``web_search`` 会在同一瞬间各自请求网络外发授权。它们是**同一个决定**，
分开问只会重复打断用户，且前端一次只呈现一个提示时，未被应答的请求
会各自等满 TTL 并最终被判拒绝，把整轮拖住。``dedupe_key`` 让同类别
的并发请求复用同一个 future：一次授权覆盖这一批（docs 03.7.1）。
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
    # 仅对 ``kind="question"`` 有意义：允许多选时前端以可切换的选项呈现答案。
    multi_select: bool = False
    future: asyncio.Future = field(default=None)  # type: ignore[assignment]
    # 正在等待该请求的调用者数量（合并请求会有多个）。归零时统一摘除，
    # 因此共享的 future 不会被某个提前超时的等待者取消掉。
    waiters: int = 0


class ConfirmBus:
    """会话作用域的请求注册表，桥接 loop 与 HTTP 客户端。"""

    def __init__(self, *, ttl_s: float = 120.0) -> None:
        self.ttl_s = ttl_s
        self._pending: dict[str, PendingRequest] = {}
        # 合并表：dedupe_key -> request_id。指向 ``_pending`` 中的首个请求，
        # 同 key 的后继调用者复用它而不重复宣告。
        self._shared: dict[str, str] = {}

    async def request(
        self,
        *,
        session_id: str,
        kind: str,
        prompt: str,
        options: list[str] | None = None,
        multi_select: bool = False,
        ttl_s: float | None = None,
        user_id: str = "",
        announce: Callable[[dict], Awaitable[None]] | None = None,
        dedupe_key: str | None = None,
    ) -> tuple[dict, str | None]:
        """向用户提问；返回 ``(request_payload, answer)``。

        ``announce`` 在请求登记之后、等待应答之前被 await，
        这样传输层可以投递提示而不与客户端的回复竞争。``answer`` 在超时时
        为 ``None``，调用方会将其呈现为显式过期，使模型知道用户从未回复。

        ``dedupe_key`` 非空时，同一 key 的并发请求合并为一个：首个调用者
        登记并宣告，其余调用者等待同一个 future 与同一个 ``request_id``，
        获得同一个答案。
        """
        timeout = ttl_s if ttl_s is not None else self.ttl_s
        if dedupe_key is not None:
            shared = self._lookup_shared(dedupe_key)
            if shared is not None:
                return await self._await_answer(shared, timeout)

        request_id = f"req_{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        pending = PendingRequest(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            kind=kind,
            prompt=prompt,
            options=list(options or []),
            multi_select=multi_select,
            future=loop.create_future(),
        )
        self._pending[request_id] = pending
        if dedupe_key is not None:
            self._shared[dedupe_key] = request_id
        if announce is not None:
            # 先登记再宣告：快速客户端可能立即应答。
            await announce(self._payload(pending))
        return await self._await_answer(pending, timeout)

    async def _await_answer(
        self, pending: PendingRequest, timeout: float
    ) -> tuple[dict, str | None]:
        """等待 ``pending`` 落定；超时返回 ``None``，被取消则向上传播。

        这里用 ``asyncio.wait`` 而非 ``asyncio.wait_for``：后者在超时时会
        取消所等待的 future，而合并请求的 future 是多个调用者共用的——
        一个等待者超时不应殃及其余仍在等待的同伴。
        """
        pending.waiters += 1
        try:
            done, _ = await asyncio.wait({pending.future}, timeout=timeout)
            if pending.future in done:
                # 被 cancel_session 取消时 result() 抛出 CancelledError，
                # 这是刻意的停止/断线语义（见模块文档与 cancel_session）。
                return self._payload(pending), str(pending.future.result())
            return self._payload(pending), None
        finally:
            pending.waiters -= 1
            if pending.waiters <= 0:
                self._release(pending)

    @staticmethod
    def _payload(pending: PendingRequest) -> dict:
        return {
            "request_id": pending.request_id,
            "kind": pending.kind,
            "prompt": pending.prompt,
            "options": pending.options,
            "multi_select": pending.multi_select,
        }

    def _lookup_shared(self, dedupe_key: str) -> PendingRequest | None:
        request_id = self._shared.get(dedupe_key)
        if request_id is None:
            return None
        pending = self._pending.get(request_id)
        # 已落定但尚未被最后一位等待者释放的条目不再复用：把旧答案发给
        # 新调用者会让一次性授权被后续调用白拿。此时登记一条新的提问。
        if pending is None or pending.future.done():
            self._shared.pop(dedupe_key, None)
            return None
        return pending

    def _release(self, pending: PendingRequest) -> None:
        """最后一个等待者离开时摘除该请求；未落定的 future 一并取消。"""
        self._pending.pop(pending.request_id, None)
        for key, request_id in list(self._shared.items()):
            if request_id == pending.request_id:
                self._shared.pop(key, None)
        if not pending.future.done():
            pending.future.cancel()

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
        # 摘除指向已取消请求的合并表项，避免后续同 key 调用命中悬空条目。
        for key, request_id in list(self._shared.items()):
            if request_id not in self._pending:
                self._shared.pop(key, None)
        return cancelled

    def pending_ids(self) -> list[str]:
        return list(self._pending)
