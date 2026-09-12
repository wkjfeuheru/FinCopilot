"""Valuation history for one symbol."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class ValuationInput(BaseModel):
    symbol: str = Field(description="6位A股代码")
    lookback_years: int = Field(default=1, description="回溯年数：1/3/5")


class GetValuationTool(BaseTool):
    name = "get_valuation"
    description = "查询A股估值走势（市值/市盈率等）及区间分位。"
    input_model = ValuationInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 30

    async def _dispatch(self, *, symbol: str, lookback_years: int = 1) -> RawData:
        return await self.data.valuation(symbol, lookback_years=lookback_years)
