"""calc_metrics：对已获取的数据做杜邦分解（文档 3.4.4）。

既接受 ``symbol``（自行获取所需数据），也接受引用会话中已有数据的 ``cids``，
因此追问的成本为零次额外取数。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, ToolGroup, param, tool
from finharness.tools.base import BaseTool

# 不同报表格式的列关键词不同，故按子串匹配。
_NET_MARGIN_KEYS = ("销售净利率", "net_margin", "net profit margin")
_ASSET_TURNOVER_KEYS = ("总资产周转率", "asset_turnover")
_EQUITY_MULTIPLIER_KEYS = ("权益乘数", "equity_multiplier")
_ROE_KEYS = ("净资产收益率", "roe")


def _find_column(df: pd.DataFrame, keys: tuple[str, ...]) -> str | None:
    """按关键词子串匹配数据框列名，返回首个匹配的列；无匹配返回 None。"""
    for column in df.columns:
        label = str(column)
        if any(key.lower() in label.lower() for key in keys):
            return column
    return None


@tool(
    name="calc_metrics",
    description="按杜邦框架分解 ROE（净利率×周转率×权益乘数），可复用已取数据。",
    capability=Capability.COMPUTE,
    group=ToolGroup.FIN_CALC,
    timeout=60,
)
class CalcMetricsTool(BaseTool):
    @param("method", desc="分析方法，当前支持 dupont")
    @param("symbol", desc="6位A股代码；与 cids 二选一")
    @param("cids", desc="复用的数据引用 id（来自先前工具结果）")
    async def _dispatch(
        self, *, method: str = "dupont", symbol: str | None = None, cids: list[str] | None = None
    ) -> RawData:
        """执行杜邦三因素分解，并按净利率、周转率、权益乘数组装结果文本。

        优先复用 `cids` 引用的已有数据，否则按 `symbol` 获取指标；返回内含
        计算文本与底层数据框的 `RawData`。支持的 `method` 目前仅 `dupont`。
        """
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

    # -- 辅助方法 --------------------------------------------------------------
    async def _resolve_frame(self, symbol: str | None, cids: list[str]) -> tuple[pd.DataFrame, str]:
        """优先复用引用数据（零额外取数）；否则获取一次指标数据。"""
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
        """读取引用所指向的 parquet 文件，避免重新取数。"""
        cite = getattr(self.ctx, "cite", None)
        if cite is None:
            return None
        for cid in cids:
            citation = cite.get(cid)
            if citation is None or not citation.parquet_path:
                continue
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
