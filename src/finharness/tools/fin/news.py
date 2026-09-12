"""Individual-stock or topic market news."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class NewsInput(BaseModel):
    symbol: str | None = Field(default=None, description="个股新闻的6位A股代码，可选")
    topic: str | None = Field(default=None, description="主题关键词（无 symbol 时使用）")
    top_n: int = Field(default=10, description="返回条数")


class GetMarketNewsTool(BaseTool):
    name = "get_market_news"
    description = "查询个股或市场主题相关新闻资讯。"
    input_model = NewsInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 30

    async def _dispatch(
        self, *, symbol: str | None = None, topic: str | None = None, top_n: int = 10
    ) -> RawData:
        return await self.data.news(symbol=symbol, topic=topic, top_n=top_n)
