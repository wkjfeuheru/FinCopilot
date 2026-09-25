"""历史 K 线，采用摘要优先的渲染方式。"""

from __future__ import annotations

from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, ToolGroup, param, tool
from finharness.tools.base import BaseTool
from finharness.tools.fin.window import note_actual_window


@tool(
    name="get_kline",
    description="查询A股或指数历史K线（日/周/月），输出区间摘要与近期明细。",
    capability=Capability.MARKET,
    group=ToolGroup.FIN_DATA,
    timeout=30,
    data_tool=True,
    output_schema_note="返回区间涨跌、极值、均线与近期明细（markdown）。",
)
class GetKlineTool(BaseTool):
    @param(
        "symbol",
        desc="6位A股代码，或指数代码（如 000300 沪深300、000905 中证500、000985 中证全指）",
    )
    @param("period", desc="周期：day/week/month")
    @param("adjust", desc="复权：qfq前复权/hfq后复权/None不复权（指数无复权概念，该参数被忽略）")
    @param("years", desc="回溯年数，如 1 表示近一年")
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
        note_actual_window(
            lines, df, raw,
            year_param="years",
            citation_note="引用时不得表述为“近 {requested} 年”，应以上述实际区间为准。",
        )
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
