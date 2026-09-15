"""历史 K 线，采用摘要优先的渲染方式。"""

from __future__ import annotations

import pandas as pd
from pydantic import Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, DataInput, PermissionLevel, ToolGroup

# 窗口长度比请求的短这么多，说明数据源已无更多数据（次新股、停牌），
# 而非序列只是“时间尚短”。
_SHORT_WINDOW_RATIO = 0.8


class KlineInput(DataInput):
    symbol: str = Field(description="6位A股代码")
    period: str = Field(default="day", description="周期：day/week/month")
    adjust: str | None = Field(default=None, description="复权：qfq前复权/hfq后复权/None不复权")
    years: int = Field(default=1, description="回溯年数，如 1 表示近一年")


class GetKlineTool(BaseTool):
    name = "get_kline"
    description = "查询A股历史K线（日/周/月），输出区间摘要与近期明细。"
    input_model = KlineInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 30
    output_schema_note = "返回区间涨跌、极值、均线与近期明细（markdown）。"

    async def _dispatch(
        self, *, symbol: str, period: str = "day", adjust: str | None = None, years: int = 1
    ) -> RawData:
        return await self.data.kline(symbol, period=period, adjust=adjust, years=years)

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """先汇总区间表现，再展示限定行数的明细表。"""
        df = raw.df
        if df is None or not len(df):
            return "（无数据）", []
        # 各数据源的行序不同；统一规范为最新在前，使摘要与均线窗口
        # 始终描述最新一期。
        if "date" in df.columns:
            df = df.sort_values("date", ascending=False).reset_index(drop=True)
        lines: list[str] = []
        self._note_actual_window(lines, df, raw)
        if "close" in df.columns:
            closes = df["close"].astype(float)
            lines.append("区间摘要：")
            lines.append(f"- 最新收盘：{closes.iloc[0]:.2f}")
            lines.append(f"- 区间最高：{closes.max():.2f} / 区间最低：{closes.min():.2f}")
            if len(closes) > 1 and closes.iloc[-1]:
                change = (closes.iloc[0] - closes.iloc[-1]) / float(closes.iloc[-1]) * 100
                lines.append(f"- 区间收益率：{change:.2f}%")
            if len(closes) >= 20:
                lines.append(f"- MA20：{closes.head(20).mean():.2f}")
            if len(closes) >= 60:
                lines.append(f"- MA60：{closes.head(60).mean():.2f}")
        # 传入完整数据框：``trim_dataframe`` 自行限定行数，且只有看到完整
        # 序列才能报告省略了多少行。
        detail = self.trim_dataframe(
            df, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        body = "\n".join(lines) + "\n\n近期明细：\n" + detail
        return body, [raw]

    @staticmethod
    def _note_actual_window(lines: list[str], df: pd.DataFrame, raw: RawData) -> None:
        """说明实际覆盖的区间，避免对过短的序列错误标注。

        对次新股调用 ``get_kline(years=N)`` 只会返回它已有的那些交易日；
        若不给出实际边界，调用方可能把这个短序列当作“近 N 年”来引用。
        该提示让这种不匹配变得明确。
        """
        if "date" not in df.columns:
            return
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not len(dates):
            return
        first, last = dates.min().date(), dates.max().date()
        lines.append(f"- 数据区间：{first} ~ {last}（共 {len(df)} 条）")
        requested = None
        try:
            requested = int((raw.params or {}).get("years"))
        except (TypeError, ValueError):
            requested = None
        if not requested or requested <= 0:
            return
        span_days = (last - first).days
        if span_days < 365 * requested * _SHORT_WINDOW_RATIO:
            lines.append(
                f"- 注意：实际区间约 {span_days} 天，明显短于请求的 {requested} 年"
                "（该标的可能上市较晚或数据不足）；引用时不得表述为"
                f"“近 {requested} 年”，应以上述实际区间为准。"
            )
