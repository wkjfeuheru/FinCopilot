"""Historical K-line with a summary-first render."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup

_SUMMARY_ROWS = 20


class KlineInput(BaseModel):
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
        """Summarize the interval, then show a bounded detail table."""
        df = raw.df
        if df is None or not len(df):
            return "（无数据）", []
        # Sources differ in row order; normalize to newest-first so the summary
        # and the MA windows always describe the latest period.
        if "date" in df.columns:
            df = df.sort_values("date", ascending=False).reset_index(drop=True)
        lines: list[str] = []
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
        detail = self.trim_dataframe(df.head(_SUMMARY_ROWS))
        body = "\n".join(lines) + "\n\n近期明细：\n" + detail
        return body, [raw]
