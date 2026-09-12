"""calc_metrics: DuPont decomposition over already-fetched data (docs 3.4.4).

Accepts either a ``symbol`` (fetches what it needs) or ``cids`` referencing data
already in the session, so a follow-up question costs zero extra fetches.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup

# Column keywords differ across report formats, so match on substrings.
_NET_MARGIN_KEYS = ("销售净利率", "net_margin", "net profit margin")
_ASSET_TURNOVER_KEYS = ("总资产周转率", "asset_turnover")
_EQUITY_MULTIPLIER_KEYS = ("权益乘数", "equity_multiplier")
_ROE_KEYS = ("净资产收益率", "roe")


def _find_column(df: pd.DataFrame, keys: tuple[str, ...]) -> str | None:
    for column in df.columns:
        label = str(column)
        if any(key.lower() in label.lower() for key in keys):
            return column
    return None


class MetricsInput(BaseModel):
    method: str = Field(default="dupont", description="分析方法，当前支持 dupont")
    symbol: str | None = Field(default=None, description="6位A股代码；与 cids 二选一")
    cids: list[str] = Field(
        default_factory=list, description="复用的数据引用 id（来自先前工具结果）"
    )


class CalcMetricsTool(BaseTool):
    name = "calc_metrics"
    description = "按杜邦框架分解 ROE（净利率×周转率×权益乘数），可复用已取数据。"
    input_model = MetricsInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_CALC
    timeout = 60

    async def _dispatch(
        self, *, method: str = "dupont", symbol: str | None = None, cids: list[str] | None = None
    ) -> RawData:
        if method != "dupont":
            raise ValueError(f"暂不支持的分析方法：{method}")
        if not symbol and not cids:
            raise ValueError("calc_metrics 需要 symbol 或 cids 之一")

        df, reused_from = await self._resolve_frame(symbol, list(cids or []))
        lines = ["杜邦分解（三因素）："]
        roe_col = _find_column(df, _ROE_KEYS)
        margin_col = _find_column(df, _NET_MARGIN_KEYS)
        turnover_col = _find_column(df, _ASSET_TURNOVER_KEYS)
        multiplier_col = _find_column(df, _EQUITY_MULTIPLIER_KEYS)

        if roe_col:
            lines.append(f"- 净资产收益率：{self._latest(df, roe_col)}%")
        if margin_col:
            lines.append(f"- 销售净利率：{self._latest(df, margin_col)}%")
        if turnover_col:
            lines.append(f"- 总资产周转率：{self._latest(df, turnover_col)}")
        if multiplier_col:
            lines.append(f"- 权益乘数：{self._latest(df, multiplier_col)}")
        if not any((roe_col, margin_col, turnover_col, multiplier_col)):
            available = "、".join(str(c) for c in df.columns[:8])
            lines.append(f"（数据中未找到杜邦分解所需字段；可用列：{available}）")
        lines.append(f"\n数据来源：{'复用 ' + reused_from if reused_from else '本次取数'}")

        return RawData(
            kind="text",
            text="\n".join(lines),
            endpoint="calc:dupont",
            params={"method": method, "symbol": symbol, "reused": reused_from},
            df=df,
        )

    # -- helpers --------------------------------------------------------------
    async def _resolve_frame(self, symbol: str | None, cids: list[str]) -> tuple[pd.DataFrame, str]:
        """Prefer reused citations (zero fetches); else fetch indicators once."""
        if cids:
            frame = self._frame_from_citations(cids)
            if frame is not None:
                return frame, ",".join(cids)
            raise ValueError(f"引用的 cid 无可用数据：{', '.join(cids)}")
        if symbol is None:
            raise ValueError("缺少 symbol")
        raw = await self.data.indicators(symbol, years=3)
        return raw.df, ""

    def _frame_from_citations(self, cids: list[str]) -> pd.DataFrame | None:
        """Load the parquet a citation points at, avoiding a refetch."""
        cite = getattr(self.ctx, "cite", None)
        if cite is None:
            return None
        for cid in cids:
            citation = cite.get(cid)
            if citation is None or not citation.parquet_path:
                continue
            from pathlib import Path

            path = Path(citation.parquet_path)
            if path.is_file():
                return pd.read_parquet(path)
        return None

    @staticmethod
    def _latest(df: pd.DataFrame, column: str) -> str:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if not len(series):
            return "-"
        return f"{series.iloc[0]:.2f}"
