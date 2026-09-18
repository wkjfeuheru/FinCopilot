"""可比公司 / 同行业估值对比。"""

from __future__ import annotations

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool


@tool(
    name="get_peers",
    description="查询可比公司/同行业估值对比（市盈率、市净率、市销率等排名）。",
    capability=Capability.PEER,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
)
class GetPeersTool(BaseTool):
    @param("symbol", desc="被比较的标的，6位A股代码")
    @param("fields", desc="字段关键词过滤，如 ['市盈率','市净率']；None 返回全部")
    async def _dispatch(self, *, symbol: str, fields: list[str] | None = None) -> RawData:
        return await self.data.peers(symbol, fields=fields)
