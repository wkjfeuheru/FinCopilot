"""单只 A 股代码的最新行情。"""

from __future__ import annotations

from pydantic import Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, DataInput, PermissionLevel, ToolGroup


class QuoteInput(DataInput):
    symbol: str = Field(description="6位A股代码，如 600519")


class GetQuoteTool(BaseTool):
    name = "get_quote"
    description = "查询单只A股的最新行情快照（价格、涨跌幅、成交量）。"
    input_model = QuoteInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 30

    async def _dispatch(self, *, symbol: str) -> RawData:
        return await self.data.quote(symbol)
