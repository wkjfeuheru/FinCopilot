"""get_macro_indicators：面向宏观研究场景的宏观经济时间序列。

只有当这些序列可用时，宏观场景才有数据支撑；在本工具出现之前，该场景
只能产出没有数字的定性文字。指标以一张长表返回
（``date | indicator | label | value | unit``），因为各序列频率不同
（月频/日频/季频），若做宽表连接，得到的将大多为空单元格。
"""

from __future__ import annotations

from pydantic import Field

from finharness.data.mapping import MACRO_INDICATORS, MACRO_INDICATOR_LABELS, normalize_macro_indicator
from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, DataInput, PermissionLevel, ToolGroup

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


class MacroInput(DataInput):
    indicators: list[str] = Field(
        default_factory=list,
        description=(
            "宏观指标名（可中文或英文 slug）。可选："
            + "、".join(f"{slug}（{spec.label}）" for slug, spec in MACRO_INDICATORS.items())
        ),
    )
    years: int = Field(default=3, description="回溯年数，如 3 表示近三年")


class GetMacroIndicatorsTool(BaseTool):
    name = "get_macro_indicators"
    description = (
        "查询中国宏观经济指标（PMI/CPI/PPI/M2/社融/LPR/SHIBOR/国债收益率/汇率/GDP），"
        "返回指标时序与最新值。用于宏观研究与宏观对资产的影响分析。"
    )
    input_model = MacroInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    timeout = 60
    output_schema_note = "返回各指标最新值摘要 + 近期时序明细（长表：日期/指标/数值/单位）。"

    async def _dispatch(self, *, indicators: list[str] | None = None, years: int = 3) -> RawData:
        """获取宏观指标序列；未指定时用默认仪表盘，并把指标名规范化为标准 slug。"""
        names = [str(x) for x in (indicators or []) if str(x).strip()]
        if not names:
            names = list(DEFAULT_INDICATORS)
        canonical = [normalize_macro_indicator(name) for name in names]
        raw = await self.data.macro(canonical, years=years)
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
        detail = self.trim_dataframe(
            df, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        body = "\n".join(lines) + "\n\n时序明细：\n" + detail
        return body, [raw]
