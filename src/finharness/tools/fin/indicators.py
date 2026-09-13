"""Financial indicator history (ROE, margins, leverage, ...)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.mapping import indicator_field_matches
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

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """Render the frame, naming any requested field the source does not have.

        A field that matches no column is a real gap (the source lacks it, or it
        is a derived metric), not an empty value; saying so keeps the reader from
        reading its absence as a data hole to be filled with an estimate.
        """
        df = raw.df
        if df is None or not len(df):
            return "（无数据）", []
        lines: list[str] = []
        requested = list((raw.params or {}).get("fields") or [])
        columns = [c for c in df.columns if str(c) != "date"]
        missing = [
            str(field)
            for field in requested
            if not any(indicator_field_matches(column, str(field)) for column in columns)
        ]
        if missing:
            lines.append(
                "注意：以下请求字段在该数据源中无对应列，未包含在结果中："
                + "、".join(missing)
                + "（缺失原因可能是该来源不提供此指标，或需由计算工具推导）；回答时不得"
                "将未返回的指标表述为已取得。"
            )
        body = self.trim_dataframe(df)
        return (("\n".join(lines) + "\n") if lines else "") + body, [raw]
