"""指定日期范围内的公司公告。"""

from __future__ import annotations

from datetime import date, timedelta

from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, Tier, ToolGroup, param, tool
from finharness.tools.base import BaseTool


def _default_since() -> str:
    return (date.today() - timedelta(days=180)).isoformat()


@tool(
    name="get_announcements",
    description="查询A股公司公告（定期报告、临时公告等）。",
    capability=Capability.ANNOUNCEMENT,
    # 大多数金融问题从不需要公告全文，而其 schema 每轮都要重发，故按需注入。
    tier=Tier.LAZY,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
)
class GetAnnouncementsTool(BaseTool):
    @param("symbol", desc="6位A股代码")
    @param(
        "since",
        desc="起始日期 YYYY-MM-DD，默认近半年",
        default_factory=_default_since,
    )
    @param("top_n", desc="返回条数")
    async def _dispatch(self, *, symbol: str, since: str, top_n: int = 20) -> RawData:
        return await self.data.announcements(symbol, since=since, top_n=top_n)
