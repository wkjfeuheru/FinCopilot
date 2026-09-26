"""行业数据：申万行业指数表现、涨跌幅排行与成分股。

两个工具共用同一条数据路由（``kind="industry"``）：
``get_industry_perf`` 回答“这个板块表现如何”，在不指定行业时返回一级行业
总览（``view="overview"``）或涨跌幅排行（``view="ranking"``）；
``get_industry_constituents`` 提供成分股，同时可作为 ``quant-factor``
的横截面股票池来源。
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, ToolGroup, param, tool
from finharness.tools.base import MAX_RENDER_ROWS, BaseTool

# 排行周期 -> 面向读者的话术。结论行用它把“单日/单周/单月”写清楚，
# 避免读者把区间收益误读成单日收益。
_PERIOD_LABELS: dict[str, str] = {"day": "单日", "week": "单周", "month": "单月"}
# 结论行最多点名多少个行业：用户问“前五”这类问题时，前五名的一句话摘要
# 就是他真正要的答案。取全量排行时它同样是一句可用的导语。
_HEADLINE_NAMES = 5


@tool(
    name="get_industry_perf",
    description=(
        "查询申万行业指数表现。给定 industry 返回其指数历史行情；省略 industry 时"
        "用 view 选择视图：view='overview'（默认）返回申万一级行业估值总览，"
        "view='ranking' 返回申万一级行业涨跌幅排行（按涨跌幅降序，一次返回全部行业）。"
        "问“涨幅前五／涨得最好／跌幅最大”这类排序问题用 view='ranking'，"
        "并用 period 指定 day/week/month、用 top 指定前若干名。"
    ),
    capability=Capability.INDUSTRY,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
    output_schema_note="返回行业指数的区间摘要与近期明细、一级行业估值总览，或一级行业涨跌幅排行。",
)
class GetIndustryPerfTool(BaseTool):
    @param(
        "industry",
        desc="申万一级行业名或代码，如 白酒/电子/801010；省略则按 view 返回总览或排行",
    )
    @param("years", desc="回溯年数（仅指定 industry 取指数历史时生效）")
    @param(
        "view",
        desc="省略 industry 时的视图：overview（默认，一级行业估值总览）或 ranking（一级行业涨跌幅排行）",
    )
    @param("period", desc="排行周期：day（默认，单日）/ week（单周）/ month（单月），仅 view='ranking' 生效")
    @param("top", desc="只取涨跌幅前若干名，如问“前五”传 5；省略返回全部一级行业")
    async def _dispatch(
        self,
        *,
        industry: str | None = None,
        years: int = 1,
        view: Literal["overview", "ranking"] = "overview",
        period: Literal["day", "week", "month"] = "day",
        top: int | None = None,
    ) -> RawData:
        if industry:
            return await self.data.industry_perf(industry, years=years)
        if view == "ranking":
            raw = await self.data.industry_ranking(period=period, as_of=None)
            # 渲染需要知道周期与条数：它们不是取数参数（不拆缓存槽位），因此挂到
            # 结果的 params 上交给渲染器，而不是塞进缓存键。
            if isinstance(raw.params, dict):
                raw.params.update({"view": "ranking", "period": period})
                if top is not None:
                    raw.params["top"] = int(top)
            return raw
        return await self.data.industry_perf(None, years=years)

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """渲染行业指数结果：排行视图自成一格，其余按有无 date 列分流。"""
        df = raw.df
        if df is None or not len(df):
            return "（无数据）", []
        if str((raw.params or {}).get("view")) == "ranking":
            return self._render_ranking(raw), [raw]
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

    def _render_ranking(self, raw: RawData) -> str:
        """渲染一级行业涨跌幅排行：先给结论行，再给已排序的表格。

        数据在适配器里已按涨跌幅降序排列，这里只做转述，不再重新排序——排版
        与排序由工具负责，模型只负责把结论说给用户，避免它在多行之间重新编排
        时把行业名与数值错配。
        """
        df = raw.df
        params: dict[str, Any] = raw.params if isinstance(raw.params, dict) else {}
        period = str(params.get("period") or "day")
        label = _PERIOD_LABELS.get(period, period)
        top = params.get("top")
        total = len(df)

        view = df.head(int(top)) if top else df.head(MAX_RENDER_ROWS)
        omitted = total - len(view)

        data_date = self._ranking_date(df, raw)
        header = f"申万一级行业{label}涨跌幅排行（数据日期：{data_date}，共 {total} 个行业）"

        records = view.to_dict("records")
        headline: list[str] = []
        for index, record in enumerate(records[:_HEADLINE_NAMES], start=1):
            name = record.get("industry") or record.get("指数名称") or record.get("行业名称")
            pct = record.get("pct_change")
            if name is None or pct is None or pd.isna(pct):
                continue
            headline.append(f"第 {index} 名 {name} {float(pct):+.2f}%")
        lines = [header]
        if headline:
            lines.append("；".join(headline))
        lines.append("")
        lines.append(self._ranking_markdown(records))
        if omitted > 0:
            shown = len(view)
            lines.append(f"（共 {total} 个一级行业，此处只列涨跌幅前 {shown} 名）")
        return "\n".join(lines)

    @staticmethod
    def _ranking_markdown(records: list[dict[str, Any]]) -> str:
        """把排好序的记录渲染成 markdown 表；数值列保留符号与两位小数。"""

        def pct(value: Any) -> str:
            return "—" if value is None or pd.isna(value) else f"{float(value):+.2f}%"

        def close(value: Any) -> str:
            return "—" if value is None or pd.isna(value) else f"{float(value):.2f}"

        rows = [
            {
                "排名": index,
                "行业代码": record.get("code") or record.get("指数代码") or "—",
                "行业名称": record.get("industry") or record.get("指数名称") or record.get("行业名称") or "—",
                "涨跌幅": pct(record.get("pct_change")),
                "收盘": close(record.get("close")),
            }
            for index, record in enumerate(records, start=1)
        ]
        return pd.DataFrame(rows).to_markdown(index=False)

    @staticmethod
    def _ranking_date(df: pd.DataFrame, raw: RawData) -> str:
        """排行的数据日期；优先用结果自带的 date 列，其次用数据时效字段。"""
        if "date" in df.columns:
            parsed = pd.to_datetime(df["date"], errors="coerce").dropna()
            if len(parsed):
                return parsed.max().date().isoformat()
        return raw.data_date or "未标注"


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
