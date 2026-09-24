"""历史序列"实际区间"提示：K 线与估值两处共用的渲染辅助。

对次新股或数据不足的标的调用 ``get_kline(years=N)`` / 估值工具，只会返回
它已有的那些交易日；若不给出实际边界，调用方可能把这个短序列当作"近 N 年"
来引用。这里把两个工具曾各自实现一遍的同一段渲染逻辑收敛到一处。
"""

from __future__ import annotations

import pandas as pd

from finharness.data.raw import RawData

# 窗口长度比请求的短这么多，说明数据源已无更多数据（次新股、停牌），
# 而非序列只是"时间尚短"。
SHORT_WINDOW_RATIO = 0.8


def note_actual_window(
    lines: list[str],
    df: pd.DataFrame,
    raw: RawData,
    *,
    year_param: str,
    citation_note: str,
) -> None:
    """向 ``lines`` 追加实际覆盖区间，并在窗口明显偏短时给出提示。

    ``year_param`` 是请求年数所在的参数字段名；``citation_note`` 是"提示的
    落脚句"模板，需含 ``{requested}`` 占位符——两个工具对引用口径的措辞不同
    （K 线要求以实际区间为准，估值要求说明区间后再谈分位），故由调用方给出。
    """
    if "date" not in df.columns:
        return
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if not len(dates):
        return
    first, last = dates.min().date(), dates.max().date()
    lines.append(f"- 数据区间：{first} ~ {last}（共 {len(df)} 条）")
    raw_requested = (raw.params or {}).get(year_param)
    requested: int | None = None
    try:
        requested = int(raw_requested) if raw_requested is not None else None
    except (TypeError, ValueError):
        requested = None
    if not requested or requested <= 0:
        return
    span_days = (last - first).days
    if span_days < 365 * requested * SHORT_WINDOW_RATIO:
        lines.append(
            f"- 注意：实际区间约 {span_days} 天，明显短于请求的 {requested} 年"
            "（该标的可能上市较晚或数据不足）；" + citation_note.format(requested=requested)
        )
