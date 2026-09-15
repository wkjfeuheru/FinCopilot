"""内存中的会话生命周期与 single-flight 保护。

*会话* 是一个执行窗口：它防止并发运行，并在 TTL 后过期。
*对话* 是记忆作用域，其存活时间长于任何会话（它存放在记忆存储中）。
将二者分离，才能让客户端在一个会话过期后恢复对话——
客户端持有的 id 是对话，而不是短暂的会话。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


class SessionBusyError(RuntimeError):
    pass


@dataclass
class ServerSession:
    session_id: str
    conversation_id: str
    user_id: str
    loop: object
    created_at: float
    last_active: float
    busy: bool = False


def new_conversation_id() -> str:
    return f"c_{uuid.uuid4().hex[:12]}"


class SessionRegistry:
    def __init__(self, loop_factory, ttl_s: int = 1800):
        self.loop_factory = loop_factory
        self.ttl_s = ttl_s
        self.sessions: dict[str, ServerSession] = {}

    async def ensure(
        self,
        session_id: str | None = None,
        *,
        conversation_id: str | None = None,
        user_id: str = "",
    ) -> ServerSession:
        """为本请求返回一个会话，必要时创建。

        解析顺序：

        1. 匹配 ``session_id`` 的活跃会话（当前执行窗口）；
        2. 已绑定到 ``conversation_id`` 的活跃会话（恢复一个
           上一个窗口仍打开的对话）；
        3. 否则新建会话，绑定到给定的 ``conversation_id`` 或
           一个新的 conversation id。

        会话句柄是全局唯一的，因此任何命中都必须同时校验
        ``user_id`` 归属——否则拿到他人 session_id 的客户端
        可以劫持其执行窗口。归属不符按“不存在”处理。
        """
        now = time.monotonic()
        self._evict_expired(now)

        if session_id:
            session = self.sessions.get(session_id)
            if session and now - session.last_active <= self.ttl_s:
                if session.user_id != user_id:
                    raise SessionBusyError("session is busy")
                if session.busy:
                    raise SessionBusyError("session is busy")
                session.last_active = now
                return session

        if conversation_id:
            existing = self.find_by_conversation(conversation_id)
            if existing is not None and existing.user_id == user_id:
                if existing.busy:
                    raise SessionBusyError("session is busy")
                existing.last_active = now
                return existing

        resolved_conversation = conversation_id or new_conversation_id()
        session_id = f"s_{uuid.uuid4().hex[:12]}"
        session = ServerSession(
            session_id=session_id,
            conversation_id=resolved_conversation,
            user_id=user_id,
            loop=self.loop_factory(session_id, resolved_conversation, user_id),
            created_at=now,
            last_active=now,
        )
        self.sessions[session_id] = session
        return session

    def find_by_conversation(self, conversation_id: str) -> ServerSession | None:
        """绑定到某对话的活跃会话（若有）。"""
        for session in self.sessions.values():
            if session.conversation_id == conversation_id:
                return session
        return None

    def _evict_expired(self, now: float) -> int:
        """丢弃超过 TTL 的会话。

        没有这一步，注册表只会不断增长：过期条目会永远留在
        dict 中，占着其 loop 与对话记录。对话持久化在
        记忆存储中，因此丢弃内存中的会话没有代价。
        """
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if not session.busy and now - session.last_active > self.ttl_s
        ]
        for session_id in expired:
            del self.sessions[session_id]
        return len(expired)

    def release(self, session: ServerSession) -> None:
        """释放会话：清除忙碌标记并更新最近活动时间。"""
        session.busy = False
        session.last_active = time.monotonic()
