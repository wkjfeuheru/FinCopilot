"""与领域无关的时钟辅助：统一时间戳的时区与精度。

七个模块各自实现过一份 ``_now``/``_now_stamp``/``_today_stamp``，秒级与
毫秒级、UTC 与本地时区混在一起。集中到这里后，每种时间戳只有一处定义，
同一类记录写入与读取的字符串因此始终可比较。
"""

from __future__ import annotations

from datetime import UTC, date, datetime


def utc_now() -> datetime:
    """当前 UTC 时刻（带时区信息）。"""
    return datetime.now(UTC)


def utc_now_iso(*, timespec: str = "seconds") -> str:
    """当前 UTC 时刻的 ISO 字符串；``timespec`` 为 ``"seconds"`` 或 ``"milliseconds"``。"""
    return datetime.now(UTC).isoformat(timespec=timespec)


def local_now_iso(*, timespec: str = "seconds") -> str:
    """当前本地时刻的 ISO 字符串（带本地时区偏移）。

    用于记录"抓取时刻"这类需要与用户所在时区一致的字段（如缓存条目的
    ``fetched_at``）。
    """
    return datetime.now().astimezone().isoformat(timespec=timespec)


def today_iso() -> str:
    """当前本地日期（``YYYY-MM-DD``）。

    注入给模型的"今天"锚点：缺少它，模型无法判断某个数据期是否即当前最新
    已发布期，只能把裸期间（如 2026-08）当成可能是过期的值。
    """
    return date.today().isoformat()
