"""Financial indicator lookup."""

from finharness.tools.base import BaseTool
from finharness.types import ToolResult


class GetIndicatorsTool(BaseTool):
    name = "get_indicators"
    description = "Get financial indicators for an A-share company."

    async def execute(self, *, symbol: str, years: int = 3, fields: list[str] | None = None) -> ToolResult:
        raw = await self.data.indicators(symbol, years, fields)
        return ToolResult(content=self.render_dataframe(raw.dataframe))
