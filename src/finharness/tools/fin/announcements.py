"""Company announcements over a date range."""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


def _default_since() -> str:
    return (date.today() - timedelta(days=180)).isoformat()


class AnnouncementsInput(BaseModel):
    symbol: str = Field(description="6位A股代码")
    since: str = Field(
        default_factory=_default_since, description="起始日期 YYYY-MM-DD，默认近半年"
    )
    top_n: int = Field(default=20, description="返回条数")


class GetAnnouncementsTool(BaseTool):
    name = "get_announcements"
    description = "查询A股公司公告（定期报告、临时公告等）。"
    input_model = AnnouncementsInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 60

    async def _dispatch(self, *, symbol: str, since: str, top_n: int = 20) -> RawData:
        return await self.data.announcements(symbol, since=since, top_n=top_n)
