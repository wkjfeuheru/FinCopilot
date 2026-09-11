"""Latest quote lookup."""

from finharness.tools.base import BaseTool
from finharness.types import ToolResult


class GetQuoteTool(BaseTool):
    name = "get_quote"
    description = "Get the latest A-share quote."

    async def execute(self, *, symbol: str) -> ToolResult:
        raw = await self.data.quote(symbol)
        return ToolResult(content=self.render_dataframe(raw.dataframe))
