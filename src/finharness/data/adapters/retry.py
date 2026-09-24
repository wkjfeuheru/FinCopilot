"""适配器内部的同步重试：对瞬时故障做有界重试。

akshare 的申万表抓取与同花顺的"数据未就绪"重试是同一形状：一个易失败的上游
调用、固定的尝试次数、两次之间睡一小段。这里统一这一形状，使"重试多少次、
退避多久"在适配器里只有一种写法。

注意边界：这里只表达**对同一个调用的重试**。``DataAccess`` 里的适配器降级链
不是重试（它在不同数据源之间前进，且内层重试的终止动作各不相同），因此不适用
本模块。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# 区分"未提供兜底值"与"兜底值就是 None"。
_MISSING: object = object()


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int,
    delay_for: Callable[[int], float],
    retry_on: type[BaseException] | tuple[type[BaseException], ...],
    exhausted: object = _MISSING,
) -> T | None:
    """调用 ``fn``；命中 ``retry_on`` 时按 ``delay_for(attempt)`` 退避重试。

    ``attempt`` 从 1 计数。尝试用尽后若显式给了 ``exhausted`` 则返回它
    （允许就是 ``None``），否则重新抛出最后一次异常——两种收尾在各调用点
    原本就不同，故显式区分。
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on:
            if attempt >= attempts:
                if exhausted is not _MISSING:
                    return exhausted  # type: ignore[return-value]
                raise
            time.sleep(delay_for(attempt))
    raise AssertionError("unreachable")  # pragma: no cover - attempts>=1 时不可达
