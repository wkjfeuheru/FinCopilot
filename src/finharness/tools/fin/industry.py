"""行业数据：申万行业指数表现与成分股。

两个工具共用同一条数据路由（``kind="industry"``）：
``get_industry_perf`` 回答“这个板块表现如何”，在不指定行业时返回一级行业
总览；``get_industry_constituents`` 提供成分股，同时可作为 ``quant-factor``
的横截面股票池来源。
"""

from __future__ import annotations

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool


@tool(
    name="get_industry_perf",
    description="查询申万行业指数表现：给定行业返回其指数历史行情，省略行业返回申万一级行业总览（含估值）。",
    capability=Capability.INDUSTRY,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
    output_schema_note="返回行业指数的区间摘要与近期明细，或一级行业总览表。",
)
class GetIndustryPerfTool(BaseTool):
    @param(
        "industry",
        desc="申万一级行业名或代码，如 白酒/电子/801010；省略则返回申万一级行业总览（估值横向对比）",
    )
    @param("years", desc="回溯年数")
    async def _dispatch(self, *, industry: str | None = None, years: int = 1) -> RawData:
        return await self.data.industry_perf(industry, years=years)

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """渲染行业指数结果：有 date 列时给出区间摘要与近期明细，否则直接渲染总览表。"""
        df = raw.df
        if df is None or not len(df):
            return "（无数据）", []
        # 总览数据框（无 date 列）：直接渲染表格。
        if "date" not in df.columns:
            return self.trim_dataframe(
                df, source_path=raw.parquet_path, detail=self._render_detail(raw)
            ), [raw]
        lines = ["行业指数区间摘要："]
        if "close" in df.columns:
            closes = df["close"].astype(float)
            lines.append(f"- 最新收盘：{closes.iloc[0]:.2f}")
            lines.append(f"- 区间最高：{closes.max():.2f} / 区间最低：{closes.min():.2f}")
            if len(closes) > 1 and closes.iloc[-1]:
                change = (closes.iloc[0] - closes.iloc[-1]) / float(closes.iloc[-1]) * 100
                lines.append(f"- 区间收益率：{change:.2f}%")
        dates = df["date"].dropna()
        if len(dates):
            lines.append(f"- 数据区间：{dates.min().date()} ~ {dates.max().date()}（共 {len(df)} 条）")
        detail = self.trim_dataframe(
            df, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        return "\n".join(lines) + "\n\n近期明细：\n" + detail, [raw]


@tool(
    name="get_industry_constituents",
    description="查询申万行业成分股（代码与名称），可作为横截面因子研究的股票池来源。",
    capability=Capability.INDUSTRY,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
    output_schema_note="返回成分股列表（证券代码/证券名称/权重）。",
)
class GetIndustryConstituentsTool(BaseTool):
    @param("industry", desc="申万一级行业名或代码，如 白酒/电子/801010")
    async def _dispatch(self, *, industry: str) -> RawData:
        return await self.data.industry_constituents(industry)
