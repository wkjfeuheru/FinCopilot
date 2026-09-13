"""make_chart: render a chart file from session data or a fresh fetch.

Data comes either from a ``cids`` reference (reusing what the session already
paid for) or from a ``symbol`` fetch. The render output is a markdown image link
plus one line of context; the image itself never enters the token budget.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.report.charting import FontUnavailableError, apply_style, resolve_cjk_font
from finharness.report.markdown import image_markdown
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup

CHART_TYPES = ("line", "bar", "candlestick")
DEFAULT_TITLE = "数据图表"


class ChartInput(BaseModel):
    type: str = Field(default="line", description="图表类型：line/bar/candlestick")
    title: str = Field(default=DEFAULT_TITLE, description="图表标题")
    symbol: str | None = Field(default=None, description="6位A股代码；与 cids 二选一")
    cids: list[str] = Field(
        default_factory=list, description="复用的数据引用 id（优先于 symbol，零重复取数）"
    )
    x: str | None = Field(default=None, description="X 轴列名，默认自动推断（date/日期）")
    y: str | None = Field(default=None, description="Y 轴列名，默认自动推断（close/value）")


class MakeChartTool(BaseTool):
    name = "make_chart"
    description = "根据会话数据或指定标的生成图表（走势/对比/K线），产出 PNG 并返回引用路径。"
    input_model = ChartInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_OUTPUT
    timeout = 60
    output_schema_note = "返回 ![title](path) 形式的图片引用。"

    async def _dispatch(
        self,
        *,
        type: str = "line",
        title: str = DEFAULT_TITLE,
        symbol: str | None = None,
        cids: list[str] | None = None,
        x: str | None = None,
        y: str | None = None,
    ) -> RawData:
        if type not in CHART_TYPES:
            raise ValueError(f"不支持的图表类型：{type}；可选 {'/'.join(CHART_TYPES)}")

        df = await self._resolve_frame(symbol, list(cids or []))
        if df is None or not len(df):
            raise ValueError("无可用于绘图的会话数据；请先取数或提供 cids")

        # Fail loudly before writing anything if Chinese labels cannot render.
        try:
            font = resolve_cjk_font()
        except FontUnavailableError as exc:
            raise ValueError(str(exc)) from exc

        path = self._chart_path(title)
        await asyncio.to_thread(self._draw, df, type, title, x, y, path, font)
        return RawData(
            kind="chart_path",
            text=image_markdown(title, str(path)),
            paths=[str(path)],
            endpoint="chart:matplotlib",
            params={"type": type, "title": title, "rows": int(len(df))},
        )

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """Charts are referenced, never inlined as image tokens (docs 3.4.4)."""
        return (raw.text or "（图表生成失败）"), [raw]

    # -- helpers --------------------------------------------------------------
    async def _resolve_frame(self, symbol: str | None, cids: list[str]) -> pd.DataFrame | None:
        if cids:
            frame = self._frame_from_citations(cids)
            if frame is not None:
                return frame
        if symbol:
            return (await self.data.kline(symbol, years=1)).df
        return None

    def _frame_from_citations(self, cids: list[str]) -> pd.DataFrame | None:
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

    def _chart_path(self, title: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)[:40] or "chart"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = Path(self.data.settings.paths.output_dir) / "charts"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{safe}_{stamp}.png"

    @staticmethod
    def _pick(df: pd.DataFrame, preferred: str | None, candidates: tuple[str, ...]) -> str | None:
        if preferred and preferred in df.columns:
            return preferred
        for name in candidates:
            if name in df.columns:
                return name
        return None

    @staticmethod
    def _last_numeric_column(df: pd.DataFrame, *, exclude: str | None) -> str | None:
        """Rightmost column convertible to numeric, ignoring the X axis column."""
        numeric = [
            column
            for column in df.columns
            if column != exclude
            and pd.to_numeric(df[column], errors="coerce").notna().any()
        ]
        return numeric[-1] if numeric else None

    def _draw(
        self,
        df: pd.DataFrame,
        chart_type: str,
        title: str,
        x: str | None,
        y: str | None,
        path: Path,
        font: str,
    ) -> None:
        apply_style(font=font)
        import matplotlib.pyplot as plt

        x_col = self._pick(df, x, ("date", "日期", "报告期", "公告日期"))
        y_col = self._pick(df, y, ("close", "value", "收盘", "最新价"))
        if y_col is None:
            # Valuation frames name their column after the indicator (for example
            # "市盈率(TTM)(倍)"), so fall back to the last plottable column.
            y_col = self._last_numeric_column(df, exclude=x_col)

        figure, axes = plt.subplots(figsize=(8, 4.5))
        try:
            if chart_type == "line" and x_col and y_col:
                axes.plot(pd.to_datetime(df[x_col]), pd.to_numeric(df[y_col], errors="coerce"),
                          marker="o", markersize=2, linewidth=1.2)
                axes.tick_params(axis="x", rotation=30)
            elif chart_type == "bar" and y_col:
                label_col = x_col or (df.columns[0] if len(df.columns) else None)
                labels = df[label_col].astype(str) if label_col else [str(i) for i in range(len(df))]
                axes.bar(labels[:20], pd.to_numeric(df[y_col], errors="coerce").head(20))
                axes.tick_params(axis="x", rotation=30)
            elif chart_type == "candlestick":
                self._draw_candles(axes, df, x_col)
            else:
                raise ValueError(
                    f"数据缺少可用列（需要 X/Y 轴列），当前列：{'、'.join(str(c) for c in df.columns[:8])}"
                )
            axes.set_title(title)
            axes.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(path, dpi=110)
        finally:
            plt.close(figure)

    @staticmethod
    def _draw_candles(axes, df: pd.DataFrame, x_col: str | None) -> None:
        required = ("open", "high", "low", "close")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"K线图缺少列：{'、'.join(missing)}")
        frame = df.tail(60).reset_index(drop=True)
        for index, row in frame.iterrows():
            up = row["close"] >= row["open"]
            color = "#c0392b" if up else "#27ae60"
            axes.vlines(index, row["low"], row["high"], color=color, linewidth=0.8)
            axes.vlines(index, min(row["open"], row["close"]), max(row["open"], row["close"]),
                        color=color, linewidth=3.2)
        axes.set_xlim(-1, len(frame))
        if x_col:
            step = max(len(frame) // 6, 1)
            axes.set_xticks(range(0, len(frame), step))
            axes.set_xticklabels(
                [str(v)[:10] for v in frame[x_col][::step]], rotation=30
            )
