"""calc_valuation：DCF 与可比公司估值（文档 3.4.4）。

假设由模型提供；工具只负责计算与呈现。它绝不臆造增长率或折现率，
也绝不给出买入或卖出结论——仅凭数字无法支撑这类结论。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup

VALUATION_METHODS = ("dcf", "comps")


class ValuationInput(BaseModel):
    method: str = Field(default="dcf", description="估值方法：dcf（绝对）或 comps（可比）")
    symbol: str = Field(description="6位A股代码")
    assumptions: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "DCF 假设：base_fcf(基期自由现金流,元)、growth_rates{悲观,中性,乐观}、"
            "wacc、terminal_growth、years(预测期年数)、shares(总股本,股)"
        ),
    )
    peer_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="comps 方法：可比组中位数，如 {pe_median, pb_median, target_eps, target_bvps}",
    )


class CalcValuationTool(BaseTool):
    name = "calc_valuation"
    description = (
        "按 DCF 或可比公司法估算价值区间与敏感性。假设须由调用方给出，"
        "工具只做计算与呈现，不替调用方下结论。"
    )
    input_model = ValuationInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_CALC
    timeout = 120

    async def _dispatch(
        self, *, method: str = "dcf", symbol: str, assumptions: dict | None = None,
        peer_metrics: dict | None = None,
    ) -> RawData:
        if method not in VALUATION_METHODS:
            raise ValueError(f"不支持的估值方法：{method}；可选 {'/'.join(VALUATION_METHODS)}")
        params = dict(assumptions or {})
        if method == "dcf":
            text = self._dcf(symbol, params)
        else:
            text = self._comps(symbol, dict(peer_metrics or {}))
        return RawData(
            kind="text", text=text, endpoint=f"calc:{method}",
            params={"method": method, "symbol": symbol},
        )

    # -- DCF -----------------------------------------------------------------
    def _dcf(self, symbol: str, assumptions: dict) -> str:
        """校验必要假设并渲染 DCF 估值表：按各情景增长率给出每股价值，附敏感性矩阵。"""
        required = ("base_fcf", "wacc", "terminal_growth", "growth_rates", "shares")
        missing = [key for key in required if key not in assumptions]
        if missing:
            raise ValueError(
                "DCF 缺少必要假设：" + "、".join(missing)
                + "。请提供 base_fcf/growth_rates{悲观,中性,乐观}/wacc/terminal_growth/shares。"
            )

        base_fcf = float(assumptions["base_fcf"])
        wacc = float(assumptions["wacc"])
        terminal_growth = float(assumptions["terminal_growth"])
        shares = float(assumptions["shares"])
        years = int(assumptions.get("years", 5))
        growth_rates = assumptions["growth_rates"]
        if not isinstance(growth_rates, dict) or not growth_rates:
            raise ValueError("growth_rates 需为映射，如 {悲观:0.05, 中性:0.10, 乐观:0.15}")

        # 当 g >= WACC 时终值公式发散；此时拒绝计算，而不是打印一个无意义的数字。
        if terminal_growth >= wacc:
            raise ValueError(
                f"永续增长率（{terminal_growth:.2%}）必须小于 WACC（{wacc:.2%}），否则终值无意义"
            )
        if shares <= 0:
            raise ValueError("shares 必须为正数")

        rows = ["DCF 估值（假设由调用方给出）：", ""]
        rows.append(f"- 基期自由现金流：{base_fcf:,.0f} 元")
        rows.append(f"- WACC：{wacc:.2%}　永续增长率：{terminal_growth:.2%}　预测期：{years} 年")
        rows.append(f"- 总股本：{shares:,.0f} 股")
        rows.append("")
        rows.append("| 情景 | 预测期增长率 | 股权价值(元) | 每股价值(元) |")
        rows.append("|---|---|---|---|")

        for label, rate in growth_rates.items():
            rate = float(rate)
            enterprise = self._dcf_value(base_fcf, rate, wacc, terminal_growth, years)
            rows.append(f"| {label} | {rate:.2%} | {enterprise:,.0f} | {enterprise / shares:,.2f} |")

        rows.append("")
        rows.append(self._sensitivity(base_fcf, growth_rates, wacc, terminal_growth, years, shares))
        rows.append("")
        rows.append("说明：以上为给定假设下的计算区间，**不构成投资建议**；"
                    "假设变动对结果影响显著，请结合 equity-research 场景的估值方法论"
                    "（references/valuation.md）核对取值依据。")
        return "\n".join(rows)

    @staticmethod
    def _dcf_value(base_fcf: float, growth: float, wacc: float, terminal_growth: float, years: int) -> float:
        """按给定假设计算现金流折现总额（元）：各年现金流增长后折现，另加永续终值。"""
        cash = base_fcf
        present = 0.0
        for year in range(1, years + 1):
            cash = cash * (1 + growth)
            present += cash / (1 + wacc) ** year
        terminal = cash * (1 + terminal_growth) / (wacc - terminal_growth)
        return present + terminal / (1 + wacc) ** years

    def _sensitivity(
        self, base_fcf: float, growth_rates: dict, wacc: float,
        terminal_growth: float, years: int, shares: float,
    ) -> str:
        """在 WACC 与居中增长率两个维度上做双向网格，输出每股价值的敏感性矩阵。"""
        central = float(growth_rates.get("中性") or next(iter(growth_rates.values())))
        wacc_axis = [wacc - 0.01, wacc, wacc + 0.01]
        growth_axis = [central - 0.02, central, central + 0.02]
        header = "| WACC＼增长率 | " + " | ".join(f"{g:.1%}" for g in growth_axis) + " |"
        separator = "|---" * (len(growth_axis) + 1) + "|"
        lines = ["敏感性矩阵（每股价值，元）：", header, separator]
        for rate in wacc_axis:
            if rate <= terminal_growth:
                continue
            cells = []
            for growth in growth_axis:
                value = self._dcf_value(base_fcf, growth, rate, terminal_growth, years)
                cells.append(f"{value / shares:,.2f}")
            lines.append(f"| {rate:.1%} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    # -- 可比公司 ---------------------------------------------------------------
    def _comps(self, symbol: str, metrics: dict) -> str:
        """校验必要输入并渲染可比公司相对估值：用可比组 PE/PB 中位数乘以目标标的分母得每股价值。"""
        required = ("pe_median", "target_eps")
        missing = [key for key in required if key not in metrics]
        if missing:
            raise ValueError(
                "comps 缺少必要输入：" + "、".join(missing)
                + "。请提供可比组中位数（pe_median/pb_median）与目标标的分母（target_eps/target_bvps）。"
            )
        lines = ["可比公司相对估值：", ""]
        lines.append(f"- 可比组 PE 中位数：{float(metrics['pe_median']):.2f}")
        pe_value = float(metrics["pe_median"]) * float(metrics["target_eps"])
        lines.append(f"- 按 PE 推算每股价值：**{pe_value:,.2f} 元**")
        if "pb_median" in metrics and "target_bvps" in metrics:
            lines.append(f"- 可比组 PB 中位数：{float(metrics['pb_median']):.2f}")
            pb_value = float(metrics["pb_median"]) * float(metrics["target_bvps"])
            lines.append(f"- 按 PB 推算每股价值：**{pb_value:,.2f} 元**")
        lines.append("")
        lines.append("说明：中位数须来自可比组（见 equity-research 场景 references/valuation.md "
                    "的可比选取标准）；亏损公司的 PE 不可用。以上**不构成投资建议**。")
        return "\n".join(lines)
