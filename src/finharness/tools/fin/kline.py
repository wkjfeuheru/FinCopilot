"""Historical K-line lookup with bounded rendering."""

from finharness.tools.base import BaseTool
from finharness.types import ToolResult


class GetKlineTool(BaseTool):
    name = "get_kline"
    description = "Get bounded historical A-share K-line data."

    async def execute(self, *, symbol: str, period: str = "day", adjust: str | None = None, years: int = 1) -> ToolResult:
        raw = await self.data.kline(symbol, period, adjust, years)
        dataframe = raw.dataframe
        summary = f"rows={len(dataframe)}\n"
        if "close" in dataframe.columns:
            summary += f"latest_close={dataframe['close'].iloc[-1]}\n"
        return ToolResult(content=summary + self.render_dataframe(dataframe))
