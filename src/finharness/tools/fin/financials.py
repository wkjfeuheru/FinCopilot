"""财务报表摘要（利润表 / 资产负债表 / 现金流量表）。"""

from __future__ import annotations

from pydantic import Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, DataInput, PermissionLevel, ToolGroup


class FinancialsInput(DataInput):
    symbol: str = Field(description="6位A股代码")
    statement: str = Field(default="利润", description="报表类型：利润/资产/现金流")
    years: int = Field(default=3, description="回溯年数")


class GetFinancialsTool(BaseTool):
    name = "get_financials"
    description = "查询A股公司财务报表摘要（营收、净利润、资产、现金流等）。"
    input_model = FinancialsInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 30

    async def _dispatch(
        self, *, symbol: str, statement: str = "利润", years: int = 3
    ) -> RawData:
        return await self.data.financials(symbol, statement=statement, years=years)
