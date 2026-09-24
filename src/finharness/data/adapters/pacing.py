"""适配器共用的调用节流。

多个数据源对突发调用会动态限流，因此两次上游调用之间要保持最小间隔。
akshare 与同花顺两个适配器曾各自实现一遍同一段"单调时钟 + 睡满差额"逻辑，
差异只在于是否需要加锁（同花顺允许并发会话共享适配器）。这里把节流本身
抽成一个纯函数，由适配器决定加不加锁。
"""

from __future__ import annotations

import time


def pace(last_call: float, seconds: float) -> float:
    """保证距上次调用至少 ``seconds`` 秒；返回本次调用后的时间戳。

    ``seconds <= 0`` 表示不节流，此时时间戳原样返回（与原实现一致）。
    """
    if seconds <= 0:
        return last_call
    elapsed = time.monotonic() - last_call
    if elapsed < seconds:
        time.sleep(seconds - elapsed)
    return time.monotonic()
