"""单个标的、单个指标的估值历史。

渲染时会附加一份派生摘要——最新值、区间、极值，以及当前值在该区间内的
分位——因为数据框在模型看到之前会被裁剪到前若干行。若由读者根据 250 行
序列中的 20 行片段自行计算分位，结果会是错的，所以由工具来计算。
"""

from __future__ import annotations

import pandas as pd

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool

# 实际区间远短于请求区间，说明数据源的历史用完了（次新股、接口变更），
# 而不是这条序列只是"年份还短"。
_SHORT_WINDOW_RATIO = 0.8
# 样本数低于此值时，分位不够稳定，需要提示：
# 寥寥几个点会让"第 90 分位"多半只是样本量造成的假象。
_MIN_PERCENTILE_SAMPLES = 60


@tool(
    name="get_valuation",
    description=(
        "查询A股单一估值指标的时间序列（默认市盈率TTM，可选市净率/市现率/总市值），"
        "并给出最新值在所选窗口内的历史分位。返回列为该指标名称与单位。"
    ),
    capability=Capability.VALUATION,
    group=ToolGroup.FIN_DATA,
    timeout=30,
    data_tool=True,
    output_schema_note="返回指标名、区间、最新值、区间分位与极值，附近期明细。",
)
class GetValuationTool(BaseTool):
    @param("symbol", desc="6位A股代码")
    @param("indicator", desc="估值指标：市盈率(TTM)/市盈率(静)/市净率/市现率/总市值")
    @param("lookback_years", desc="回溯年数：1/3/5")
    async def _dispatch(
        self, *, symbol: str, indicator: str = "市盈率(TTM)", lookback_years: int = 1
    ) -> RawData:
        return await self.data.valuation(
            symbol, lookback_years=lookback_years, indicator=indicator
        )

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """概括序列（最新值、区间、极值、分位），随后给出明细。

        分位回答的是"对这只股票来说当前贵不贵"，这是单看水平值无法回答的；
        分位严格在请求的区间上计算，并且会说明该区间，
        使这个数字绝不会脱离依据被引用。
        """
        df = raw.df
        if df is None or not len(df):
            return "（无数据）", []
        if "date" in df.columns:
            df = df.sort_values("date", ascending=False).reset_index(drop=True)
        column = self._value_column(df, raw)
        if column is None:
            return self.trim_dataframe(
                df, source_path=raw.parquet_path, detail=self._render_detail(raw)
            ), [raw]

        lines: list[str] = [f"估值指标：{column}"]
        self._note_window(lines, df, raw)
        self._note_percentile(lines, df, column)
        # 传入完整数据框：``trim_dataframe`` 自身会限制行数，
        # 且只有看到整条序列，它才能报告省略了多少行。
        detail = self.trim_dataframe(
            df, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        body = "\n".join(lines) + "\n\n近期明细：\n" + detail
        return body, [raw]

    # -- 辅助函数 --------------------------------------------------------------
    @staticmethod
    def _value_column(df: pd.DataFrame, raw: RawData) -> str | None:
        """在数据源与缓存两种形态下定位指标列。

        当前的数据框按指标命名该列（``市盈率(TTM)(倍)``）；
        在该变更之前缓存的数据框则只有一个裸 ``value``。返回的名称会原样
        用作标签，使摘要始终报告数据框实际持有的指标。
        """
        indicator = str((raw.params or {}).get("indicator") or "").strip()
        if indicator:
            for name in df.columns:
                if str(name).startswith(indicator):
                    return name
        if "value" in df.columns:
            return "value"
        numeric = [
            name
            for name in df.columns
            if str(name) != "date" and pd.to_numeric(df[name], errors="coerce").notna().any()
        ]
        return numeric[-1] if len(numeric) == 1 else None

    @staticmethod
    def _note_window(lines: list[str], df: pd.DataFrame, raw: RawData) -> None:
        """说明实际覆盖的区间，避免短序列被重新贴上标签。"""
        if "date" not in df.columns:
            return
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not len(dates):
            return
        first, last = dates.min().date(), dates.max().date()
        lines.append(f"- 数据区间：{first} ~ {last}（共 {len(df)} 条）")
        requested = None
        try:
            requested = int((raw.params or {}).get("lookback_years"))
        except (TypeError, ValueError):
            requested = None
        if not requested or requested <= 0:
            return
        span_days = (last - first).days
        if span_days < 365 * requested * _SHORT_WINDOW_RATIO:
            lines.append(
                f"- 注意：实际区间约 {span_days} 天，明显短于请求的 {requested} 年"
                "（该标的可能上市较晚或数据不足）；引用分位时须说明实际区间，"
                f"不得表述为“近 {requested} 年分位”。"
            )

    @staticmethod
    def _note_percentile(lines: list[str], df: pd.DataFrame, column: str) -> None:
        """报告最新值在本区间内的分位。

        分位 = 在实际返回的行中，取值小于等于最新值的观测占比。
        最新值为非正数时分位没有意义（负的 PE 并不代表"便宜"），
        因此这种情况只作说明而不计算。
        """
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if not len(series):
            lines.append("- 区间统计：该指标在区间内无数值。")
            return
        latest = float(series.iloc[0])
        lines.append(f"- 最新值：{latest:,.2f}")
        lines.append(
            f"- 区间最低：{series.min():,.2f} / 中位数：{series.median():,.2f} / "
            f"最高：{series.max():,.2f}"
        )
        if latest <= 0:
            lines.append(
                "- 区间分位：不适用（最新值为非正数，负值/零值的估值分位没有意义，"
                "不得据此判断高低）。"
            )
            return
        percentile = float((series <= latest).sum()) / len(series) * 100
        lines.append(
            f"- 区间分位：{percentile:.1f}%（最新值在所选区间 {len(series)} 个样本中的位置；"
            "数值越高表示当前值相对自身历史越高，估值类指标越高即越贵）"
        )
        if len(series) < _MIN_PERCENTILE_SAMPLES:
            lines.append(
                f"- 注意：样本仅 {len(series)} 条，分位稳定性有限，引用时应说明样本量。"
            )
