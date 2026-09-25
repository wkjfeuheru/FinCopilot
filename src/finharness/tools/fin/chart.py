"""make_chart：由会话数据或新取数据渲染图表文件。

数据可来自 ``cids`` 引用（复用会话已取过的数据），也可来自 ``symbol`` 取数。
一次调用绘制一张图：``line`` 可叠加多条数值序列（例如策略净值与其基准共用
一图），``bar`` 可将多条序列分组，``candlestick`` 绘制 OHLC。

多个 ``cids`` 会作为多条序列叠加，并按各自的 x 列对齐：这是比较同一会话
内所取两个标的估值序列的自然做法。渲染输出为 markdown 图片链接加一行上下文
说明；图片本身从不进入 token 预算。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd

from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, ToolGroup, param, tool
from finharness.tools.base import BaseTool
from finharness.tools.fin.charting import (
    FontUnavailableError,
    apply_style,
    resolve_cjk_font,
)
from finharness.utils.markdown import image_markdown

CHART_TYPES = ("line", "bar", "candlestick")
DEFAULT_TITLE = "数据图表"
# 柱状图限制行数：避免 300 行的序列渲染出 300 根无法辨认的柱子。
MAX_BAR_ROWS = 20
_X_CANDIDATES = ("date", "日期", "报告期", "公告日期", "月份", "季度")
_Y_CANDIDATES = ("close", "value", "nav", "收盘", "最新价", "净值")


@tool(
    name="make_chart",
    description=(
        "根据会话数据或指定标的生成图表（走势/多序列对比/分组柱/K线），产出 PNG 并返回引用路径。"
        "支持多序列叠加（如回测净值与基准同图）。"
    ),
    capability=Capability.OUTPUT,
    group=ToolGroup.FIN_OUTPUT,
    timeout=60,
    output_schema_note="返回 ![title](path) 形式的图片引用。",
)
class MakeChartTool(BaseTool):
    @param("type", desc="图表类型：line/bar/candlestick")
    @param("title", desc="图表标题")
    @param("symbol", desc="6位A股代码或指数代码（如 000300 沪深300）；与 cids 二选一")
    @param(
        "cids",
        desc="复用的数据引用 id；传多个时按 X 轴对齐叠加为多条序列（零重复取数）",
    )
    @param("x", desc="X 轴列名，默认自动推断（date/日期）")
    @param("y", desc="Y 轴列名（单序列），默认自动推断（close/value）")
    @param(
        "series",
        desc="要绘制的数值列（多序列）；优先级高于 y。如回测净值传 [nav, benchmark_nav]",
    )
    async def _dispatch(
        self,
        *,
        type: str = "line",
        title: str = DEFAULT_TITLE,
        symbol: str | None = None,
        cids: list[str] | None = None,
        x: str | None = None,
        y: str | None = None,
        series: list[str] | None = None,
    ) -> RawData:
        """图表入口：解析数据与列，校验字体后渲染 PNG 并返回引用。

        参数 type 取值 line/bar/candlestick；数据来自 cids 引用或 symbol 取数；
        返回 kind="chart_path" 的 RawData，文本为 markdown 图片链接。
        """
        if type not in CHART_TYPES:
            raise ValueError(f"不支持的图表类型：{type}；可选 {'/'.join(CHART_TYPES)}")

        df, overlay_series = await self._resolve(symbol, list(cids or []))
        if df is None or not len(df):
            raise ValueError("无可用于绘图的会话数据；请先取数或提供 cids")
        df = df.copy()
        numeric = self._numeric_columns(df)
        if not numeric:
            raise ValueError(f"数据中没有可绘制的数值列，当前列：{'、'.join(str(c) for c in df.columns)}")

        x_col = self._pick(df, x, _X_CANDIDATES) or (str(df.columns[0]) if overlay_series else None)

        if type == "candlestick":
            y_cols: list[str] = []
        else:
            y_cols = self._select_series(df, numeric, series=list(series or []), y=y, overlay=overlay_series)

        # 若中文字体无法渲染，则在写出任何东西之前显式报错。
        try:
            font = resolve_cjk_font()
        except FontUnavailableError as exc:
            raise ValueError(str(exc)) from exc

        path = self._chart_path(title)
        await asyncio.to_thread(self._draw, df, type, title, x_col, y_cols, path, font)
        return RawData(
            kind="chart_path",
            text=image_markdown(title, str(path)),
            paths=[str(path)],
            endpoint="chart:matplotlib",
            params={"type": type, "title": title, "rows": int(len(df)), "series": y_cols},
        )

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """图表以引用方式呈现，绝不作为图片 token 内联（docs 3.4.4）。"""
        return (raw.text or "（图表生成失败）"), [raw]

    # -- 数据解析 --------------------------------------------------------------
    async def _resolve(
        self, symbol: str | None, cids: list[str]
    ) -> tuple[pd.DataFrame | None, list[str]]:
        """返回 (frame, overlay_series)。

        ``overlay_series`` 是多个 cid 叠加时为待绘制列取的名称；单帧路径下
        为空，以便调用方回退到自动推断。
        """
        frames = [self._frame_from_cid(cid) for cid in cids]
        frames = [frame for frame in frames if frame is not None]
        if frames:
            if len(frames) == 1:
                return frames[0], []
            overlaid = self._align_overlay(cids, frames)
            if overlaid is not None:
                return overlaid, [str(c) for c in overlaid.columns if str(c) != "date"]
            # 对齐失败（无共享 x 列）；回退到第一个数据帧。
            return frames[0], []
        if symbol:
            raw = await self.data.kline(symbol, years=1)
            return raw.df, []
        return None, []

    def _frame_from_cid(self, cid: str) -> pd.DataFrame | None:
        cite = getattr(self.ctx, "cite", None)
        if cite is None:
            return None
        citation = cite.get(cid)
        if citation is None or not citation.parquet_path:
            return None
        path = Path(citation.parquet_path)
        if not path.is_file():
            return None
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError):
            return None

    def _align_overlay(
        self, cids: list[str], frames: list[pd.DataFrame]
    ) -> pd.DataFrame | None:
        """按共享的 x 列，为每个 cid 拼接一条数值序列。

        序列名称在已知时取引用的标的代码，否则取 cid，使图例显示 "600519"
        而非恰好冲突的列名。
        """
        cite = getattr(self.ctx, "cite", None)
        prepared: list[pd.DataFrame] = []
        names: list[str] = []
        for cid, frame in zip(cids, frames):
            numeric = self._numeric_columns(frame)
            if not numeric:
                continue
            x_col = self._pick(frame, None, _X_CANDIDATES)
            if x_col is None:
                continue
            y_col = self._pick(frame, None, _Y_CANDIDATES) or numeric[-1]
            name = None
            if cite is not None:
                citation = cite.get(cid)
                name = citation.symbol if citation is not None else None
            name = str(name or cid)
            block = pd.DataFrame(
                {
                    "date": pd.to_datetime(frame[x_col], errors="coerce"),
                    name: pd.to_numeric(frame[y_col], errors="coerce"),
                }
            ).dropna(subset=["date"])
            if not len(block):
                continue
            prepared.append(block.set_index("date")[name])
            names.append(name)
        if len(prepared) < 2:
            return None
        joined = pd.concat(prepared, axis=1)
        joined = joined.loc[:, ~joined.columns.duplicated()]
        if joined.dropna(how="all").empty:
            return None
        joined = joined.sort_index().reset_index()
        return joined

    # -- 列选择 ----------------------------------------------------------------
    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> list[str]:
        return [
            c
            for c in df.columns
            if pd.to_numeric(df[c], errors="coerce").notna().any()
        ]

    def _select_series(
        self,
        df: pd.DataFrame,
        numeric: list[str],
        *,
        series: list[str],
        y: str | None,
        overlay: list[str],
    ) -> list[str]:
        """确定要绘制的列，对显式请求做严格校验并在无效时报错。"""
        if series:
            self._require_columns(df, series)
            return series
        if y:
            self._require_columns(df, [y])
            return [y]
        if overlay:
            present = [c for c in overlay if c in numeric]
            if present:
                return present
        preferred = self._pick(df, None, _Y_CANDIDATES)
        if preferred in numeric:
            return [preferred]
        return [numeric[-1]]

    @staticmethod
    def _require_columns(df: pd.DataFrame, requested: list[str]) -> None:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(
                f"数据中不存在列：{'、'.join(missing)}；可用列：{'、'.join(str(c) for c in df.columns)}"
            )
        for column in requested:
            if pd.to_numeric(df[column], errors="coerce").notna().sum() == 0:
                raise ValueError(f"列 {column} 不是数值列，无法绘图")

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

    # -- 绘图 ------------------------------------------------------------------
    def _draw(
        self,
        df: pd.DataFrame,
        chart_type: str,
        title: str,
        x_col: str | None,
        y_cols: list[str],
        path: Path,
        font: str,
    ) -> None:
        """按图表类型分派到蜡烛/柱状/折线绘制，设置标题网格后保存 PNG。"""
        apply_style(font=font)
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(8, 4.5))
        try:
            if chart_type == "candlestick":
                self._draw_candles(axes, df, x_col)
            elif chart_type == "bar":
                self._draw_bars(axes, df, x_col, y_cols)
            else:
                self._draw_lines(axes, df, x_col, y_cols)
            axes.set_title(title)
            axes.grid(True, alpha=0.3)
            if chart_type != "candlestick" and len(y_cols) > 1:
                axes.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(path, dpi=110)
        finally:
            plt.close(figure)

    @staticmethod
    def _draw_lines(axes, df: pd.DataFrame, x_col: str | None, y_cols: list[str]) -> None:
        """在给定坐标轴上绘制一条或多条折线序列。"""
        if not y_cols:
            raise ValueError("折线图缺少可绘制的数值列")
        x_values = pd.to_datetime(df[x_col]) if x_col else range(len(df))
        for column in y_cols:
            axes.plot(
                x_values,
                pd.to_numeric(df[column], errors="coerce"),
                marker="o",
                markersize=2,
                linewidth=1.2,
                label=str(column),
            )
        axes.tick_params(axis="x", rotation=30)

    @staticmethod
    def _draw_bars(axes, df: pd.DataFrame, x_col: str | None, y_cols: list[str]) -> None:
        """绘制柱状图；多序列时并排分组，且仅取前 ``MAX_BAR_ROWS`` 行。"""
        if not y_cols:
            raise ValueError("柱状图缺少可绘制的数值列")
        view = df.head(MAX_BAR_ROWS).reset_index(drop=True)
        label_col = x_col or (str(df.columns[0]) if len(df.columns) else None)
        labels = view[label_col].astype(str) if label_col else [str(i) for i in range(len(view))]
        positions = range(len(view))
        if len(y_cols) == 1:
            axes.bar(list(positions), pd.to_numeric(view[y_cols[0]], errors="coerce"))
        else:
            width = 0.8 / len(y_cols)
            for index, column in enumerate(y_cols):
                offset = (index - (len(y_cols) - 1) / 2) * width
                axes.bar(
                    [p + offset for p in positions],
                    pd.to_numeric(view[column], errors="coerce"),
                    width=width,
                    label=str(column),
                )
        axes.set_xticks(list(positions))
        axes.set_xticklabels(labels, rotation=30)

    @staticmethod
    def _draw_candles(axes, df: pd.DataFrame, x_col: str | None) -> None:
        """绘制最近 60 根 K 线（红涨绿跌），影线加实体。"""
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
