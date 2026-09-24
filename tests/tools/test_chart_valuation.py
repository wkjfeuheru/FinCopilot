"""make_chart 与 calc_valuation，以及 skill 注册表的新语义。"""

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry, fingerprint_frame
from finharness.tools.fin.chart import MakeChartTool
from finharness.tools.fin.charting import (
    CJK_FONT_CANDIDATES,
    FontUnavailableError,
    resolve_cjk_font,
)
from finharness.tools.fin.valuation_calc import CalcValuationTool
from tests.conftest import settings_with_cache


class Ctx:
    def __init__(self, cite=None):
        self.cite = cite or CitationRegistry()
        self.loaded_skills: list[str] = []
        self.loaded_tools: list[str] = []


def make_settings(tmp_path) -> Settings:
    return settings_with_cache(tmp_path, paths={"output_dir": tmp_path / "output"})


def kline_frame(rows: int = 90) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "open": [100 + i for i in range(rows)],
        "high": [101 + i for i in range(rows)],
        "low": [99 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
    })


def access_with_cited_frame(tmp_path, frame: pd.DataFrame) -> tuple[DataAccess, Ctx, str]:
    settings = make_settings(tmp_path)
    parquet = tmp_path / "cache" / "kline.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet)
    cite = CitationRegistry()
    cid = cite.register(
        tool="get_kline", endpoint="akshare:stock_zh_a_daily", symbol="600519",
        params={}, rows=len(frame), cols=len(frame.columns),
        fingerprint=fingerprint_frame(frame), parquet_path=str(parquet),
    ).cid
    return DataAccess([], cache=None, settings=settings), Ctx(cite), cid


# --- 字体 ---------------------------------------------------------------------

def test_font_candidates_cover_major_platforms():
    joined = " ".join(CJK_FONT_CANDIDATES)
    assert "SimHei" in joined          # Windows
    assert "PingFang SC" in joined     # macOS
    assert "Noto Sans CJK SC" in joined  # Linux


def test_resolved_font_is_a_known_candidate():
    """宿主可能缺少 CJK 字体；若存在可用字体，则必须来自该列表。"""
    try:
        font = resolve_cjk_font()
    except FontUnavailableError as exc:
        assert "中文字体" in str(exc)
    else:
        assert font in CJK_FONT_CANDIDATES


# --- make_chart --------------------------------------------------------------

@pytest.mark.parametrize("chart_type", ["line", "candlestick", "bar"])
def test_chart_types_render_a_png(tmp_path, chart_type):
    access, ctx, cid = access_with_cited_frame(tmp_path, kline_frame())
    tool = MakeChartTool(access, ctx=ctx)

    result = asyncio.run(tool.run(type=chart_type, title="测试图", cids=[cid]))

    assert result.ok is True, result.error
    assert result.content.startswith("![测试图](")
    assert result.attachments, "a chart file must be reported as an attachment"
    assert Path(result.attachments[0]).is_file()


def test_chart_reuses_cited_parquet_without_fetching(tmp_path):
    access, ctx, cid = access_with_cited_frame(tmp_path, kline_frame())
    # 未配置任何 adapter，因此抓取会失败；成功即证明发生了复用。
    result = asyncio.run(MakeChartTool(access, ctx=ctx).run(type="line", cids=[cid]))

    assert result.ok is True


def test_unsupported_chart_type_is_refused(tmp_path):
    access, ctx, cid = access_with_cited_frame(tmp_path, kline_frame())

    result = asyncio.run(MakeChartTool(access, ctx=ctx).run(type="pie", cids=[cid]))

    assert result.ok is False
    assert "不支持的图表类型" in result.error


def test_chart_without_data_is_refused(tmp_path):
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())

    result = asyncio.run(MakeChartTool(access, ctx=ctx).run(type="line"))

    assert result.ok is False
    assert "无可用于绘图的会话数据" in result.error


def test_candlestick_requires_ohlc_columns(tmp_path):
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=5), "close": range(5)})
    access, ctx, cid = access_with_cited_frame(tmp_path, frame)

    result = asyncio.run(MakeChartTool(access, ctx=ctx).run(type="candlestick", cids=[cid]))

    assert result.ok is False
    assert "K线图缺少列" in result.error


# --- 多序列与叠加 -------------------------------------------------------------

def nav_frame(rows: int = 120) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "nav": [1 + i * 0.001 for i in range(rows)],
        "benchmark_nav": [1 + i * 0.0005 for i in range(rows)],
    })


def test_line_overlays_several_series(tmp_path):
    """回测的策略与基准必须共用同一张图。"""
    access, ctx, cid = access_with_cited_frame(tmp_path, nav_frame())
    result = asyncio.run(
        MakeChartTool(access, ctx=ctx).run(
            type="line", title="净值", cids=[cid], series=["nav", "benchmark_nav"]
        )
    )

    assert result.ok is True, result.error
    assert result.sources[0].params["series"] == ["nav", "benchmark_nav"]


def test_bar_groups_several_series(tmp_path):
    frame = pd.DataFrame({"name": ["A", "B", "C"], "pe": [20, 25, 18], "pb": [3, 4, 2]})
    access, ctx, cid = access_with_cited_frame(tmp_path, frame)

    result = asyncio.run(
        MakeChartTool(access, ctx=ctx).run(
            type="bar", title="对比", cids=[cid], series=["pe", "pb"]
        )
    )

    assert result.ok is True, result.error


def test_unknown_series_column_is_refused_with_the_available_list(tmp_path):
    access, ctx, cid = access_with_cited_frame(tmp_path, nav_frame())

    result = asyncio.run(
        MakeChartTool(access, ctx=ctx).run(type="line", cids=[cid], series=["nope"])
    )

    assert result.ok is False
    assert "不存在列" in result.error
    assert "nav" in result.error  # 会列出可用的列名


def test_single_y_still_works_alongside_series(tmp_path):
    """向后兼容：y 参数保持不变。"""
    access, ctx, cid = access_with_cited_frame(tmp_path, nav_frame())

    result = asyncio.run(MakeChartTool(access, ctx=ctx).run(type="line", cids=[cid], y="nav"))

    assert result.ok is True, result.error


def test_several_cids_are_overlaid_and_aligned(tmp_path):
    """同一会话中拉取的两个标的序列可以共处一张图。"""
    settings = make_settings(tmp_path)
    cite = CitationRegistry()

    def cite_frame(name: str, values: list[float]) -> str:
        frame = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=len(values), freq="D"),
            "close": values,
        })
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path)
        return cite.register(
            tool="get_valuation", endpoint="e", symbol=name, params={},
            rows=len(frame), cols=2, fingerprint=fingerprint_frame(frame),
            parquet_path=str(path),
        ).cid

    cid_a = cite_frame("600519", [20 + i * 0.1 for i in range(60)])
    cid_b = cite_frame("000858", [15 + i * 0.05 for i in range(60)])
    access = DataAccess([], cache=None, settings=settings)

    result = asyncio.run(
        MakeChartTool(access, ctx=Ctx(cite)).run(type="line", title="PE对比", cids=[cid_a, cid_b])
    )

    assert result.ok is True, result.error
    plotted = result.sources[0].params["series"]
    assert plotted == ["600519", "000858"]


def test_candlestick_ignores_series(tmp_path):
    """K 线由 OHLC 定义，而非由 series 参数定义。"""
    access, ctx, cid = access_with_cited_frame(tmp_path, kline_frame())

    result = asyncio.run(
        MakeChartTool(access, ctx=ctx).run(type="candlestick", cids=[cid], series=["close"])
    )

    assert result.ok is True, result.error


# --- calc_valuation ----------------------------------------------------------

def valuation_inputs() -> dict:
    return {
        "base_fcf": 1.0e10,
        "growth_rates": {"悲观": 0.03, "中性": 0.08, "乐观": 0.12},
        "wacc": 0.09,
        "terminal_growth": 0.03,
        "shares": 1.0e9,
        "years": 5,
    }


def test_dcf_produces_three_scenarios_and_a_sensitivity_grid(tmp_path):
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())
    tool = CalcValuationTool(access, ctx=ctx)

    result = asyncio.run(tool.run(method="dcf", symbol="600519", assumptions=valuation_inputs()))

    assert result.ok is True, result.error
    assert "悲观" in result.content and "乐观" in result.content
    assert "敏感性矩阵" in result.content


def test_dcf_requires_assumptions(tmp_path):
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())

    result = asyncio.run(
        CalcValuationTool(access, ctx=ctx).run(
            method="dcf", symbol="600519", assumptions={"base_fcf": 1.0}
        )
    )

    assert result.ok is False
    assert "缺少必要假设" in result.error


def test_dcf_refuses_terminal_growth_above_wacc(tmp_path):
    """g >= WACC 会使终值失去意义；应拒绝计算而非输出结果。"""
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())
    assumptions = valuation_inputs()
    assumptions["terminal_growth"] = 0.12  # 高于 wacc 0.09

    result = asyncio.run(
        CalcValuationTool(access, ctx=ctx).run(method="dcf", symbol="600519", assumptions=assumptions)
    )

    assert result.ok is False
    assert "必须小于 WACC" in result.error


def test_comps_uses_provided_medians(tmp_path):
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())

    result = asyncio.run(
        CalcValuationTool(access, ctx=ctx).run(
            method="comps", symbol="600519",
            peer_metrics={"pe_median": 25.0, "target_eps": 60.0, "pb_median": 8.0, "target_bvps": 200.0},
        )
    )

    assert result.ok is True, result.error
    assert "1,500.00" in result.content  # 25 * 60
    assert "1,600.00" in result.content  # 8 * 200


def test_comps_requires_median_and_denominator(tmp_path):
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())

    result = asyncio.run(
        CalcValuationTool(access, ctx=ctx).run(method="comps", symbol="600519", peer_metrics={})
    )

    assert result.ok is False
    assert "comps 缺少必要输入" in result.error


def test_valuation_reports_no_conclusion_of_its_own(tmp_path):
    """该工具只呈现数字，不得告诉读者买入或卖出。"""
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())

    result = asyncio.run(
        CalcValuationTool(access, ctx=ctx).run(method="dcf", symbol="600519", assumptions=valuation_inputs())
    )

    assert "不构成投资建议" in result.content
    for banned in ("建议买入", "建议卖出", "推荐买入", "推荐卖出"):
        assert banned not in result.content
