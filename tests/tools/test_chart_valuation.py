"""make_chart and calc_valuation, plus the skill registry's new semantics."""

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry, fingerprint_frame
from finharness.report.charting import CJK_FONT_CANDIDATES, FontUnavailableError, resolve_cjk_font
from finharness.tools.fin.chart import MakeChartTool
from finharness.tools.fin.valuation_calc import CalcValuationTool


class Ctx:
    def __init__(self, cite=None):
        self.cite = cite or CitationRegistry()
        self.loaded_skills: list[str] = []
        self.loaded_tools: list[str] = []


def make_settings(tmp_path) -> Settings:
    return Settings(
        paths={"output_dir": tmp_path / "output"}, data={"cache_dir": tmp_path / "cache"}
    )


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


# --- fonts -------------------------------------------------------------------

def test_font_candidates_cover_major_platforms():
    joined = " ".join(CJK_FONT_CANDIDATES)
    assert "SimHei" in joined          # Windows
    assert "PingFang SC" in joined     # macOS
    assert "Noto Sans CJK SC" in joined  # Linux


def test_resolved_font_is_a_known_candidate():
    """The host may lack CJK fonts; if it has one it must come from the list."""
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
    # No adapters are configured, so a fetch would fail; success proves reuse.
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
    """g >= WACC makes the terminal value meaningless; refuse rather than print."""
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())
    assumptions = valuation_inputs()
    assumptions["terminal_growth"] = 0.12  # above wacc 0.09

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
    """The tool presents numbers; it must not tell the reader to buy or sell."""
    access, ctx, _ = access_with_cited_frame(tmp_path, kline_frame())

    result = asyncio.run(
        CalcValuationTool(access, ctx=ctx).run(method="dcf", symbol="600519", assumptions=valuation_inputs())
    )

    assert "不构成投资建议" in result.content
    for banned in ("建议买入", "建议卖出", "推荐买入", "推荐卖出"):
        assert banned not in result.content
