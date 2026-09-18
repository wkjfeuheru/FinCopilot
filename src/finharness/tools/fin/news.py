"""个股或主题相关的市场新闻。"""

from __future__ import annotations

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool


@tool(
    name="get_market_news",
    description="查询个股或市场主题相关新闻资讯。",
    capability=Capability.NEWS,
    group=ToolGroup.FIN_DATA,
    timeout=30,
    data_tool=True,
)
class GetMarketNewsTool(BaseTool):
    @param("symbol", desc="个股新闻的6位A股代码，可选")
    @param("topic", desc="主题关键词（无 symbol 时使用）")
    @param("top_n", desc="返回条数")
    async def _dispatch(
        self, *, symbol: str | None = None, topic: str | None = None, top_n: int = 10
    ) -> RawData:
        return await self.data.news(symbol=symbol, topic=topic, top_n=top_n)
