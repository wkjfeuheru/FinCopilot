"""Valuation history for one symbol and one indicator."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class ValuationInput(BaseModel):
    symbol: str = Field(description="6位A股代码")
    indicator: str = Field(
        default="市盈率(TTM)",
        description="估值指标：市盈率(TTM)/市盈率(静)/市净率/市现率/总市值",
    )
    lookback_years: int = Field(default=1, description="回溯年数：1/3/5")


class GetValuationTool(BaseTool):
    name = "get_valuation"
    description = (
        "查询A股单一估值指标的时间序列（默认市盈率TTM，可选市净率/市现率/总市值）。"
        "返回列为该指标名称与单位，请注意区分指标口径。"
    )
    input_model = ValuationInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 30

    async def _dispatch(
        self, *, symbol: str, indicator: str = "市盈率(TTM)", lookback_years: int = 1
    ) -> RawData:
        return await self.data.valuation(
            symbol, lookback_years=lookback_years, indicator=indicator
        )
