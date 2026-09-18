"""数据时效元数据：把「这是截至哪一期的数据」变成可证明的事实（docs 03.4.2）。

存在的问题：工具结果只给数字与一个裸期间（如 ``2026-08``），读者无从判断它是不是
*最新可得的* 一期，也无从区分「数据本身按月发布、当期尚未发布」与「系统取到了旧数据」。
当用户问「最新」而回答恰好落在上一期时，这两者会被混为一谈。

本模块把时效做成结构化元数据而非一句声明：

* :class:`SeriesSpec` —— 一条序列的来源属性（频率、发布机构、发布节奏）；
* :class:`SeriesFreshness` —— 一条序列本次返回的最新数据期，连同上述来源属性；
* :class:`Freshness` —— 整次取数的时效视图（多条序列 + 抓取时刻 + 是否缓存命中），
  并能渲染成模型可读的段落。

它是通用件：宏观序列按指标分组，财报、公告、研报同样可以按报告期分组复用同一套
渲染，因此时效披露不因工具而异。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# 频率 slug → 读者可读的中文标签。
_FREQUENCY_LABELS: dict[str, str] = {
    "daily": "日度",
    "weekly": "周度",
    "monthly": "月度",
    "quarterly": "季度",
    "yearly": "年度",
}


def frequency_label(frequency: str) -> str:
    """把频率 slug 转成中文标签；未知 slug 原样返回。"""
    return _FREQUENCY_LABELS.get(str(frequency).strip().lower(), str(frequency))


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """一条序列的时效来源属性。

    这些属性是**关于该指标如何发布**的静态事实（机构、频率、节奏），与本次取到的
    具体数值无关，因此可以按指标一次性声明、多处复用。
    """

    frequency: str = ""
    publisher: str = ""
    cadence: str = ""


@dataclass(frozen=True, slots=True)
class SeriesFreshness:
    """一条序列本次返回的最新数据期，连同它的来源属性。"""

    label: str
    period_iso: str
    frequency: str = ""
    publisher: str = ""
    cadence: str = ""

    @property
    def period_display(self) -> str:
        """面向读者的数据期写法。

        月频写 ``2026年8月``，季频写 ``2026年1季度``，日频保留完整日期——读者据此
        对齐口径，不必去猜 ``2026-08`` 指的是整月还是某一天。
        """
        text = str(self.period_iso)
        try:
            parts = text[:10].split("-")
            year, month = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return text
        frequency = str(self.frequency).strip().lower()
        if frequency == "yearly":
            return f"{year}年"
        if frequency == "quarterly":
            return f"{year}年{(month - 1) // 3 + 1}季度"
        if frequency == "monthly":
            return f"{year}年{month}月"
        day = parts[2] if len(parts) > 2 else ""
        return f"{year}-{month:02d}-{day}" if day else f"{year}-{month:02d}"

    def describe(self) -> str:
        """单行描述：``制造业PMI 2026年8月（月度，中国物流与采购联合会，当月最后一日发布）``。"""
        qualifiers = [part for part in (frequency_label(self.frequency), self.publisher) if part]
        cadence = f"，{self.cadence}" if self.cadence else ""
        suffix = f"（{'，'.join(qualifiers)}{cadence}）" if qualifiers or cadence else ""
        return f"{self.label} {self.period_display}{suffix}"


@dataclass(frozen=True, slots=True)
class Freshness:
    """整次取数的时效视图。

    ``fetched_at`` 是数据实际被取回的时刻（缓存命中时为该条目写入缓存的时刻），
    因此读者既能看见数据期，也能看见这份数据在本地停留了多久。
    """

    series: tuple[SeriesFreshness, ...] = ()
    fetched_at: str = ""
    from_cache: bool = False

    def __bool__(self) -> bool:
        return bool(self.series)

    @property
    def latest_period_iso(self) -> str:
        """所有序列中最新的一条数据期（用于快速比较排序）。"""
        periods = [item.period_iso for item in self.series if item.period_iso]
        return max(periods) if periods else ""

    def note(self) -> str:
        """渲染成模型可读的段落；无序列时返回空串。

        每一行给出「指标 → 数据期 → 发布机构/频率/节奏」，末尾给出抓取时刻与是否
        缓存命中。刻意只陈述事实：数据期是否即当前最新已发布期，由模型结合当前日期
        与发布节奏判断，工具不替它下这个结论。
        """
        if not self.series:
            return ""
        lines = ["数据时效（本次取数各序列的最新数据期）："]
        lines.extend(f"- {item.describe()}" for item in self.series)
        when = f"；数据抓取时刻 {self.fetched_at}" if self.fetched_at else ""
        origin = "本地缓存命中，未重新联网取数" if self.from_cache else "本次实时取数"
        lines.append(f"以上为本次取数返回的最新数据期{when}（{origin}）。")
        return "\n".join(lines)


def freshness_from_grouped_frame(
    df: Any,
    *,
    group_col: str,
    period_col: str,
    label_col: str | None = None,
    value_col: str | None = None,
    meta: Mapping[str, SeriesSpec] | None = None,
    fetched_at: str = "",
    from_cache: bool = False,
) -> Freshness:
    """从长表派生时效元数据：每个分组取它最新的一条非空数据期。

    按 ``group_col`` 分组（如宏观长表的 ``indicator``），使每条序列各自报告自己的
    最新期——若只看整表最大日期，日频序列会把月频序列的期数掩盖过去。

    分组标识到来源属性的映射由 ``meta`` 给出；缺省时该序列只报数据期。
    """
    if df is None or not len(df):
        return Freshness()
    if period_col not in df.columns or group_col not in df.columns:
        return Freshness()

    import pandas as pd

    working = df
    if value_col and value_col in working.columns:
        working = working.dropna(subset=[value_col])

    series: list[SeriesFreshness] = []
    for group, block in working.groupby(group_col, sort=False):
        periods = pd.to_datetime(block[period_col], errors="coerce").dropna()
        if not len(periods):
            continue
        latest = periods.max()
        key = str(group)
        label = key
        if label_col and label_col in block.columns and len(block):
            label = str(block[label_col].iloc[0])
        spec = meta.get(key) if meta else None
        series.append(
            SeriesFreshness(
                label=label,
                period_iso=latest.date().isoformat(),
                frequency=spec.frequency if spec else "",
                publisher=spec.publisher if spec else "",
                cadence=spec.cadence if spec else "",
            )
        )
    if not series:
        return Freshness()
    return Freshness(series=tuple(series), fetched_at=fetched_at, from_cache=from_cache)
