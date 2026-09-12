"""Financial indicator history (ROE, margins, leverage, ...)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class IndicatorsInput(BaseModel):
    symbol: str = Field(description="6位A股代码")
    years: int = Field(default=3, description="回溯年数")
    fields: list[str] | None = Field(
        default=None, description="字段关键词过滤，如 ['ROE','毛利率']；None 返回全部"
    )


class GetIndicatorsTool(BaseTool):
    name = "get_indicators"
    description = "查询A股财务指标历史（盈利能力、成长性、偿债能力等）。"
    input_model = IndicatorsInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 30

    async def _dispatch(
        self, *, symbol: str, years: int = 3, fields: list[str] | None = None
    ) -> RawData:
        return await self.data.indicators(symbol, years=years, fields=fields)
