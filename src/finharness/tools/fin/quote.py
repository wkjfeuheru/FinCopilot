"""单只 A 股代码的最新行情。"""

from __future__ import annotations

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool


@tool(
    name="get_quote",
    description="查询单只A股的最新行情快照（价格、涨跌幅、成交量）。",
    capability=Capability.MARKET,
    group=ToolGroup.FIN_DATA,
    timeout=30,
    data_tool=True,
)
class GetQuoteTool(BaseTool):
    @param("symbol", desc="6位A股代码，如 600519")
    async def _dispatch(self, *, symbol: str) -> RawData:
        return await self.data.quote(symbol)
