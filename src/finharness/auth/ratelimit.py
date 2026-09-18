"""进程内固定窗口限速：认证端点与每租户用量预算（隔离方案 P0-5/P0-6）。

多租户之前，这里什么都不需要：只有一个使用者，限速的唯一作用是挡住自己。
一旦服务暴露给互不信任的用户，缺了它就有两类后果——**认证端点**可被用来
暴力猜密码与批量注册，**对话端点**可被单个租户用来耗尽进程、上游数据源与
模型配额，让其他租户一起变慢。

设计取舍：

* **固定窗口而非令牌桶。** 需要的是"挡住滥用"，而不是精确整形；固定窗口的
  实现与计数语义（窗口内第 N 次起拒绝）都简单到可以被测试断言，而令牌桶的
  平滑性在这里换不来什么。
* **进程内状态。** 与缓存/会话注册表一致：单进程部署下正确，多副本部署下
  每个副本各自计数（见模块末尾的边界说明）。不引入 Redis，避免把可用性
  依赖于一个本可不存在的组件。
* **有界。** 计数器用 ``BoundedMap`` 封顶，否则"按 IP 限速"会变成一个
  攻击者可远程撑大的内存表——限速器自己成了 DoS 面。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from finharness.utils.bounded import BoundedMap

__all__ = ["RateLimitDecision", "RateLimiter"]


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """一次限速判定：``allowed`` 为 False 时附上恢复所需秒数。"""

    allowed: bool
    remaining: int = 0
    retry_after_s: int = 0


@dataclass(slots=True)
class _Window:
    started_at: float
    count: int = 0


class RateLimiter:
    """固定窗口计数器。

    ``limit <= 0`` 表示不限制，使调用方可以沿用"未配置即关闭"的约定，
    也让默认配置不改变单用户本地行为。
    """

    def __init__(
        self,
        *,
        limit: int,
        window_s: float,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = int(limit)
        self.window_s = float(window_s)
        # 有界：键来自 IP / user_id，二者都是攻击者可控的。
        self._windows: BoundedMap[str, _Window] = BoundedMap(max_size=max_keys)

    @property
    def enabled(self) -> bool:
        return self.limit > 0 and self.window_s > 0

    def check(self, key: str) -> RateLimitDecision:
        """记一次使用并判定是否放行。"""
        if not self.enabled:
            return RateLimitDecision(allowed=True, remaining=self.limit)

        now = time.monotonic()
        window = self._windows.peek(key)
        if window is None or now - window.started_at >= self.window_s:
            self._windows[key] = _Window(started_at=now, count=1)
            return RateLimitDecision(allowed=True, remaining=self.limit - 1)

        if window.count >= self.limit:
            elapsed = now - window.started_at
            retry_after = max(1, int(self.window_s - elapsed) + 1)
            return RateLimitDecision(allowed=False, remaining=0, retry_after_s=retry_after)

        window.count += 1
        return RateLimitDecision(allowed=True, remaining=self.limit - window.count)

    def reset(self, key: str) -> None:
        """清空某个键的计数（例如登录成功后）。"""
        self._windows.pop(key, None)

    def prune(self) -> int:
        """丢弃已过期的窗口；返回丢弃数量。

        窗口本来就会在下次访问时滚动，因此这不是必需的；它存在的意义是让长驻
        进程在"大量键各自只被访问过一次"之后能主动回收，而不必等 LRU 挤出。
        """
        now = time.monotonic()
        stale = [
            key
            for key, window in list(self._windows.items())
            if now - window.started_at >= self.window_s
        ]
        for key in stale:
            self._windows.pop(key, None)
        return len(stale)


# 边界说明（写入此处以免被误读为"限速已完成"）：以上计数是**进程内**的。
# 多副本部署时每个副本各有一份计数，因此有效上限是 limit × 副本数；要得到
# 全局上限需要共享存储（Redis 等），那属于部署形态的决策，不在这里替运维方
# 决定。同样地，按 IP 限速在反向代理后需要真实客户端 IP（X-Forwarded-For），
# 否则所有用户共享同一个代理地址——见 api.py 中 `_client_key` 的处理。
