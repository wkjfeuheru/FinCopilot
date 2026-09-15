"""get_valuation 的 render：序列摘要、窗口诚实性与分位边界。

分位由 tool 而非模型计算，因为 frame 在到达模型之前会被裁剪；
从多年期序列的 20 行摘录中读出的分位会是错误的。这些测试锁定数值与守卫。
"""

import asyncio

import pandas as pd

from finharness.config.settings import ContextSettings, Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.data.raw import RawData
from finharness.tools.fin.valuation import GetValuationTool

INDICATOR = "市盈率(TTM)"


def make_settings(tmp_path) -> Settings:
    return Settings(
        context=ContextSettings(max_result_tokens=1000, trim_rows=20),
        data={"cache_dir": tmp_path / "cache"},
    )


def make_tool(tmp_path) -> GetValuationTool:
    return GetValuationTool(DataAccess([], cache=None, settings=make_settings(tmp_path)))


def valuation_raw(df: pd.DataFrame, *, indicator: str = INDICATOR, years: int = 1) -> RawData:
    return RawData(
        kind="df",
        df=df,
        endpoint="akshare:stock_zh_valuation_baidu",
        params={"symbol": "600519", "indicator": indicator, "lookback_years": years},
        data_date="2026-09-12",
    )


def daily_series(values: list[float]) -> pd.DataFrame:
    """最新在前的日频 frame，与 adapter 的顺序一致。"""
    dates = pd.date_range("2026-09-12", periods=len(values), freq="-1D")
    return pd.DataFrame({"date": dates, "市盈率(TTM)(倍)": values})


def test_percentile_is_the_share_of_observations_at_or_below_the_latest(tmp_path):
    # 最新在前：最新值 50 是最大值，因此位于第 100 分位。
    df = daily_series([50.0, 40.0, 30.0, 20.0, 10.0])

    body, _ = make_tool(tmp_path).render(valuation_raw(df))

    assert "最新值：50.00" in body
    assert "区间分位：100.0%" in body


def test_percentile_of_a_midpoint_value(tmp_path):
    df = daily_series([30.0, 50.0, 10.0, 40.0, 20.0])

    body, _ = make_tool(tmp_path).render(valuation_raw(df))

    # 10、20 与 30 均 <= 30 -> 3/5。
    assert "区间分位：60.0%" in body


def test_summary_names_the_metric_and_its_unit(tmp_path):
    df = daily_series([20.0, 21.0, 22.0])

    body, _ = make_tool(tmp_path).render(valuation_raw(df))

    assert "估值指标：市盈率(TTM)(倍)" in body
    assert "区间最低：20.00" in body
    assert "最高：22.00" in body


def test_negative_latest_reports_no_percentile(tmp_path):
    """负的 PE 不算"便宜"，因此不提供分位。"""
    df = daily_series([-5.0, -3.0, 12.0, 20.0])

    body, _ = make_tool(tmp_path).render(valuation_raw(df))

    assert "区间分位：不适用" in body
    assert "非正数" in body
    assert "%" not in body.split("区间分位")[1].split("(")[0]


def test_thin_sample_is_flagged(tmp_path):
    df = daily_series([20.0, 21.0, 22.0])

    body, _ = make_tool(tmp_path).render(valuation_raw(df))

    assert "样本仅 3 条" in body


def test_window_shorter_than_requested_is_flagged(tmp_path):
    df = daily_series([20.0, 21.0, 22.0])

    body, _ = make_tool(tmp_path).render(valuation_raw(df, years=3))

    assert "明显短于请求的 3 年" in body
    assert "近 3 年分位" in body


def test_full_window_has_no_short_window_warning(tmp_path):
    df = daily_series([20.0 + (i % 5) for i in range(400)])

    body, _ = make_tool(tmp_path).render(valuation_raw(df, years=1))

    assert "数据区间" in body
    assert "明显短于" not in body


def test_legacy_value_column_is_still_summarized(tmp_path):
    """在列被命名之前缓存的 frame 只带有一个裸 ``value``。"""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-09-12", periods=3, freq="-1D"),
            "value": [15.0, 14.0, 13.0],
        }
    )

    body, _ = make_tool(tmp_path).render(valuation_raw(df))

    assert "最新值：15.00" in body
    assert "区间分位：100.0%" in body


def test_empty_frame_is_reported_not_crashed(tmp_path):
    body, sources = make_tool(tmp_path).render(valuation_raw(pd.DataFrame()))

    assert body == "（无数据）"
    assert sources == []


def test_output_names_the_actual_window(tmp_path):
    df = daily_series([20.0, 21.0])

    body, _ = make_tool(tmp_path).render(valuation_raw(df))

    assert "数据区间：" in body
    assert "共 2 条" in body


class _ValuationAdapter(DataAdapter):
    name = "fake"

    def fetch_valuation(self, symbol, lookback_years, indicator):
        return FetchResult(
            df=pd.DataFrame(
                {
                    "date": pd.date_range("2026-09-12", periods=3, freq="-1D"),
                    "市盈率(TTM)(倍)": [20.0, 19.0, 18.0],
                }
            ),
            interface="valuation_api",
        )


def test_tool_end_to_end_carries_the_percentile(tmp_path):
    access = DataAccess(
        [_ValuationAdapter()],
        cache=LocalCache(tmp_path / "cache"),
        settings=make_settings(tmp_path),
    )

    result = asyncio.run(
        GetValuationTool(access).run(symbol="600519", indicator=INDICATOR, lookback_years=1)
    )

    assert result.ok is True
    assert "区间分位：100.0%" in result.content
