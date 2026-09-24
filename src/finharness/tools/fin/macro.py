"""get_macro_indicators：面向宏观研究场景的宏观经济时间序列。

只有当这些序列可用时，宏观场景才有数据支撑；在本工具出现之前，该场景
只能产出没有数字的定性文字。指标以一张长表返回
（``date | indicator | label | value | unit``），因为各序列频率不同
（月频/日频/季频），若做宽表连接，得到的将大多为空单元格。
"""

from __future__ import annotations

from finharness.data.freshness import SeriesSpec, freshness_from_grouped_frame
from finharness.data.mapping import (
    MACRO_INDICATOR_LABELS,
    MACRO_INDICATORS,
    normalize_macro_indicator,
)
from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, ToolGroup, param, tool
from finharness.tools.base import BaseTool

# 一套完整的宏观仪表盘；模型会将其收窄到问题实际需要的指标。
DEFAULT_INDICATORS = (
    "pmi_manufacturing",
    "cpi_yoy",
    "ppi_yoy",
    "m2_yoy",
    "social_financing",
    "lpr_1y",
    "bond_10y",
)

# 时序明细里每条序列展示的最近期数。日频序列（国债收益率、汇率、SHIBOR）与月频序列
# 同表堆叠，早期版本对整个长表做一次 head()，结果被日频序列占满，月频指标一行都进不了
# 模型。按序列各自取最近若干期，使月频指标也拿得到上月对比值。
_PER_SERIES_ROWS = 6
# 频率 slug → 用于排序的粗糙粒度，使明细里同频序列聚在一起、且月频不会排在日频之前
# 而被挤掉。仅影响展示顺序，不改变数据。
_FREQUENCY_RANK = {"monthly": 0, "quarterly": 1, "weekly": 2, "daily": 3}

_INDICATOR_HELP = (
    "宏观指标名（可中文或英文 slug）。可选："
    + "、".join(f"{slug}（{spec.label}）" for slug, spec in MACRO_INDICATORS.items())
)


@tool(
    name="get_macro_indicators",
    description=(
        "查询中国宏观经济指标（PMI/CPI/PPI/M2/社融/LPR/SHIBOR/国债收益率/汇率/GDP），"
        "返回指标时序与最新值。用于宏观研究与宏观对资产的影响分析。"
    ),
    capability=Capability.MACRO,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
    output_schema_note="返回各指标最新值摘要 + 近期时序明细（长表：日期/指标/数值/单位）。",
)
class GetMacroIndicatorsTool(BaseTool):
    @param("indicators", desc=_INDICATOR_HELP)
    @param("years", desc="回溯年数，如 3 表示近三年")
    async def _dispatch(self, *, indicators: list[str] | None = None, years: int = 3) -> RawData:
        """获取宏观指标序列；未指定时用默认仪表盘，并把指标名规范化为标准 slug。"""
        names = [str(x) for x in (indicators or []) if str(x).strip()]
        if not names:
            names = list(DEFAULT_INDICATORS)
        canonical = [normalize_macro_indicator(name) for name in names]
        raw = await self.data.macro(canonical, years=years)
        # 时效元数据在这里由数据框派生：每条序列报自己的最新数据期（见 freshness.py），
        # 而不是整表最大日期——后者会被日频序列的最近交易日顶到当天，对月频指标没有意义。
        spec_by_slug = {
            slug: SeriesSpec(
                frequency=spec.frequency,
                publisher=spec.publisher,
                cadence=spec.cadence,
            )
            for slug, spec in MACRO_INDICATORS.items()
        }
        raw.freshness = freshness_from_grouped_frame(
            raw.df,
            group_col="indicator",
            period_col="date",
            label_col="label",
            value_col="value",
            meta=spec_by_slug,
            fetched_at=str(raw.fetched_at or ""),
            from_cache=bool(raw.from_cache),
        )
        return raw

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """渲染宏观结果：先给出各指标最新值摘要，再附近期时序明细（长表）。"""
        df = raw.df
        if df is None or not len(df):
            return "（无数据）", []
        lines: list[str] = []
        if {"indicator", "value", "date"} <= set(df.columns):
            lines.append("最新值：")
            for slug, block in df.groupby("indicator", sort=False):
                latest = block.dropna(subset=["value"]).head(1)
                if not len(latest):
                    continue
                row = latest.iloc[0]
                label = MACRO_INDICATOR_LABELS.get(str(slug), str(slug))
                period = str(row["date"])[:7] if hasattr(row["date"], "strftime") else str(row["date"])
                unit = row.get("unit", "")
                lines.append(f"- {label}：{float(row['value']):.2f}{unit}（{period}）")
        detail = self.trim_dataframe_grouped(
            self._ordered_for_display(df),
            group_col="indicator",
            per_group_rows=_PER_SERIES_ROWS,
            source_path=raw.parquet_path,
            detail=self._render_detail(raw),
        )
        body = "\n".join(lines) + "\n\n时序明细：\n" + detail
        return body, [raw]

    @staticmethod
    def _ordered_for_display(df):
        """把低频序列排在前面，使它们在分组裁剪与阅读顺序上都先出场。

        排序键为（频率粗糙序，指标 slug），因此同频序列相邻、月频/季频不会被日频序列
        的开头淹没。数据框本身已按指标、日期降序排列，本函数只调整组间顺序。
        """
        if "indicator" not in df.columns:
            return df
        ranks = {slug: _FREQUENCY_RANK.get(spec.frequency, 9) for slug, spec in MACRO_INDICATORS.items()}
        order = {slug: index for index, slug in enumerate(df["indicator"].drop_duplicates().tolist())}
        keyed = df.assign(
            _rank=df["indicator"].map(lambda slug: ranks.get(str(slug), 9)),
            _order=df["indicator"].map(lambda slug: order.get(slug, 0)),
        )
        keyed = keyed.sort_values(["_rank", "_order"], kind="stable")
        return keyed.drop(columns=["_rank", "_order"])
