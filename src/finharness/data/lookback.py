"""取数窗口的起始日期计算。

适配器在多个位置重复书写 ``date.today() - timedelta(days=365 * years + N)``：
一律用 365 天近似一年，并加一个余量天数以覆盖交易日与接口对齐的偏差。
集中到一处后，余量口径只有一份，避免各接口的窗口悄悄漂移。
"""

from __future__ import annotations

from datetime import date, timedelta


def lookback_start(years: int, *, extra_days: int = 0) -> date:
    """``years`` 年前的今天，额外回退 ``extra_days`` 天作为安全余量。"""
    return date.today() - timedelta(days=365 * max(years, 1) + extra_days)


def lookback_stamp(years: int, *, extra_days: int = 0) -> str:
    """``lookback_start`` 的 ``YYYYMMDD`` 形式（多数接口的日期入参格式）。"""
    return lookback_start(years, extra_days=extra_days).strftime("%Y%m%d")
