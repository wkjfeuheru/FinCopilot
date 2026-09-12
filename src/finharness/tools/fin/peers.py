"""Peer / comparable-company valuation comparison."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class PeersInput(BaseModel):
    symbol: str = Field(description="被比较的标的，6位A股代码")
    fields: list[str] | None = Field(
        default=None, description="字段关键词过滤，如 ['市盈率','市净率']；None 返回全部"
    )


class GetPeersTool(BaseTool):
    name = "get_peers"
    description = "查询可比公司/同行业估值对比（市盈率、市净率、市销率等排名）。"
    input_model = PeersInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 60

    async def _dispatch(self, *, symbol: str, fields: list[str] | None = None) -> RawData:
        return await self.data.peers(symbol, fields=fields)
