"""内存中的会话生命周期与 single-flight 保护。

*会话* 是一个执行窗口：它防止并发运行，并在 TTL 后过期。
*对话* 是记忆作用域，其存活时间长于任何会话（它存放在记忆存储中）。
将二者分离，才能让客户端在一个会话过期后恢复对话——
客户端持有的 id 是对话，而不是短暂的会话。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from finharness.types import StopSignal
from finharness.compute.executor import ComputeExecutor, ComputeResult, ComputeTask
from collections.abc import Mapping


@dataclass(frozen=True)
class SessionCompute:
    """服务端绑定会话身份，业务任务只提供类型与输入。"""
    executor: ComputeExecutor
    user_id: str
    conversation_id: str

    async def execute(self, *, kind: str, files: Mapping[str, bytes], timeout_s: float = 120) -> ComputeResult:
        return await self.executor.execute(ComputeTask(
            self.user_id, self.conversation_id, kind, files, timeout_s=timeout_s
        ))


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
    # 进入 busy 的时刻；用于区分"在飞请求"与"被遗弃的请求"。
    # None 表示尚未记录（mark_busy 之前的旧状态），按在飞处理。
    busy_since: float | None = None
    # 当前在飞请求的停止信号（docs 03.3）。挂在会话上而非 loop 上，因为
    # 停止端点在 loop 之外运行，需要一条够得到它的路径；请求结束时清空。
    stop_signal: StopSignal | None = field(default=None)
    compute: SessionCompute | None = None


def new_conversation_id() -> str:
    return f"c_{uuid.uuid4().hex[:12]}"


class SessionRegistry:
    def __init__(self, loop_factory, ttl_s: int = 1800, busy_timeout_s: int | None = None,
                 compute_executor: ComputeExecutor | None = None):
        self.loop_factory = loop_factory
        self.ttl_s = ttl_s
        self.compute_executor = compute_executor
        # 一个 busy 会话正常也会在若干秒内结束；持续 busy 远超这个时长只可能是
        # 请求被遗弃（流被丢弃、异常逃逸导致 ``release`` 没跑到）。此类条目若
        # 永不回收，就会永久占着一个 AgentLoop 与它的整条记忆。
        self.busy_timeout_s = (
            int(busy_timeout_s) if busy_timeout_s is not None else max(2 * ttl_s, ttl_s + 600)
        )
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
            compute=SessionCompute(self.compute_executor, user_id, resolved_conversation)
            if self.compute_executor is not None else None,
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
        """丢弃超过 TTL 的会话，以及 busy 状态已持续到不可能仍在运行的会话。

        若没有这一步，注册表只会不断增长：过期条目会永远留在 dict 中，占着
        其 loop 与对话记录。对话持久化在记忆存储中，因此丢弃内存中的会话
        没有代价。

        在飞的请求（``busy`` 且刚开始）**不会**被回收——那会让同一对话并发
        写入。只有 busy 持续超过 ``busy_timeout_s`` 的条目才视为被遗弃。
        """
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if self._is_stale(session, now)
        ]
        for session_id in expired:
            del self.sessions[session_id]
        return len(expired)

    def _is_stale(self, session: ServerSession, now: float) -> bool:
        """会话是否可以被安全丢弃。"""
        idle = now - session.last_active
        if not session.busy:
            return idle > self.ttl_s
        # busy 会话：只有在**已知** busy 起始时刻、且已持续到离谱时才回收，
        # 避免误伤在飞请求。缺少起始时刻时按在飞处理（宁可留着也不误删）。
        if session.busy_since is None:
            return False
        return now - session.busy_since > self.busy_timeout_s

    def release(self, session: ServerSession) -> None:
        """释放会话：清除忙碌标记并更新最近活动时间。"""
        session.busy = False
        session.busy_since = None
        # 清掉停止信号：它属于刚结束的那次请求。留着它会让下一次请求一开始
        # 就处于"已请求停止"的状态。
        session.stop_signal = None
        session.last_active = time.monotonic()
        # 一轮结束是回收过期会话的自然时机：否则空闲进程要等到下一个请求
        # 才会触发淘汰，过期会话会一直占着内存。
        self._evict_expired(session.last_active)

    def mark_busy(self, session: ServerSession) -> None:
        """把会话标记为正在处理，并记下起始时刻供遗弃检测使用。"""
        session.busy = True
        session.busy_since = time.monotonic()

    def find(self, *, session_id: str | None, conversation_id: str | None) -> ServerSession | None:
        """按 session_id（优先）或 conversation_id 定位活跃会话。

        刻意绕开 ``ensure``：后者会把 busy 会话判为 409，而停止端点恰恰只对
        busy 会话有意义。
        """
        if session_id:
            session = self.sessions.get(session_id)
            if session is not None:
                return session
        if conversation_id:
            return self.find_by_conversation(conversation_id)
        return None

    def busy_count(self, user_id: str) -> int:
        """该用户当前在飞的会话数。

        用于每租户的并发上限（隔离方案 P0-6）。刻意从注册表**现有状态**推导，
        而不维护一个独立的计数器：计数器需要在每条退出路径上递减，而异常、
        断线与停止这几条路径恰恰最容易漏减，一旦漏了就永久挤占该租户的额度。
        这里的读法没有这个失效模式。
        """
        return sum(
            1
            for session in self.sessions.values()
            if session.busy and session.user_id == user_id
        )
