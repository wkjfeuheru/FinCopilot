"""财务指标历史（ROE、利润率、杠杆率等）。"""

from __future__ import annotations

from finharness.data.mapping import indicator_field_matches
from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, ToolGroup, param, tool
from finharness.tools.base import BaseTool


@tool(
    name="get_indicators",
    description="查询A股财务指标历史（盈利能力、成长性、偿债能力等）。",
    capability=Capability.FINANCIAL,
    group=ToolGroup.FIN_DATA,
    timeout=30,
    data_tool=True,
)
class GetIndicatorsTool(BaseTool):
    @param("symbol", desc="6位A股代码")
    @param("years", desc="回溯年数")
    @param("fields", desc="字段关键词过滤，如 ['ROE','毛利率']；None 返回全部")
    async def _dispatch(
        self, *, symbol: str, years: int = 3, fields: list[str] | None = None
    ) -> RawData:
        return await self.data.indicators(symbol, years=years, fields=fields)

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """渲染数据框，并标注数据源中不存在的所请求字段。

        匹配不到任何列的字段是真实缺口（数据源不提供该指标，或它是由计算
        推导的指标），而非空值；明确指出这一点，可避免读者把字段缺失误读
        为需要用估算值填补的数据空洞。
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
        body = self.trim_dataframe(
            df, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        return (("\n".join(lines) + "\n") if lines else "") + body, [raw]
