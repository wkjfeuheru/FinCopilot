"""财务报表摘要（利润表 / 资产负债表 / 现金流量表）。"""

from __future__ import annotations

from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, ToolGroup, param, tool
from finharness.tools.base import BaseTool


@tool(
    name="get_financials",
    description="查询A股公司财务报表摘要（营收、净利润、资产、现金流等）。",
    capability=Capability.FINANCIAL,
    group=ToolGroup.FIN_DATA,
    timeout=30,
    data_tool=True,
)
class GetFinancialsTool(BaseTool):
    @param("symbol", desc="6位A股代码")
    @param("statement", desc="报表类型：利润/资产/现金流")
    @param("years", desc="回溯年数")
    async def _dispatch(
        self, *, symbol: str, statement: str = "利润", years: int = 3
    ) -> RawData:
        return await self.data.financials(symbol, statement=statement, years=years)
