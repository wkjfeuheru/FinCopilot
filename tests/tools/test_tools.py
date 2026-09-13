import asyncio

import pandas as pd

from finharness.config.settings import ContextSettings, Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.tools.fin.kline import GetKlineTool
from finharness.tools.fin.quote import GetQuoteTool


class Adapter(DataAdapter):
    name = "fake"

    def fetch_quote(self, symbol):
        return FetchResult(
            df=pd.DataFrame([{"symbol": symbol, "close": 100.0}]), interface="fake_quote"
        )

    def fetch_kline(self, symbol, period, adjust, years):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "close": [100.0 + i for i in range(30)],
                "open": [99.0 + i for i in range(30)],
                "high": [101.0 + i for i in range(30)],
                "low": [98.0 + i for i in range(30)],
            }
        )
        return FetchResult(df=df, interface="fake_kline")


def make_settings(**kwargs) -> Settings:
    return Settings(context=ContextSettings(**kwargs))


def make_access(adapters, tmp_path, **kwargs) -> DataAccess:
    """Hermetic access: an explicit cache keeps runs out of the repo cache."""
    return DataAccess(adapters, cache=LocalCache(tmp_path / "cache"), settings=make_settings(**kwargs))


def test_quote_tool_renders_adapter_data(tmp_path):
    async def run():
        return await GetQuoteTool(make_access([Adapter()], tmp_path)).run(symbol="600519")

    result = asyncio.run(run())

    assert result.ok is True
    assert "100" in result.content
    assert result.sources and result.sources[0].endpoint == "fake:fake_quote"


def test_kline_tool_renders_a_summary_and_detail(tmp_path):
    async def run():
        return await GetKlineTool(make_access([Adapter()], tmp_path)).run(
            symbol="600519", period="day", adjust=None, years=1
        )

    result = asyncio.run(run())

    assert result.ok is True
    assert "区间摘要" in result.content
    assert "MA20" in result.content


def test_kline_summary_uses_the_latest_row_not_the_first(tmp_path):
    """Sources return either row order; the render must read the newest close."""

    class AscendingAdapter(DataAdapter):
        name = "asc"

        def fetch_kline(self, symbol, period, adjust, years):
            # Ascending order, as Sina returns it.
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-06-01", "2026-09-11"]),
                    "close": [100.0, 200.0, 300.0],
                }
            )
            return FetchResult(df=df, interface="asc_kline")

    async def run():
        return await GetKlineTool(make_access([AscendingAdapter()], tmp_path)).run(
            symbol="600519", period="day", adjust=None, years=1
        )

    result = asyncio.run(run())

    assert result.ok is True
    # The newest close (300.0) must be reported, not the oldest.
    assert "300.00" in result.content


def test_kline_rejects_invalid_symbol(tmp_path):
    async def run():
        return await GetKlineTool(make_access([Adapter()], tmp_path)).run(
            symbol="bad", period="day", adjust=None, years=1
        )

    result = asyncio.run(run())

    assert result.ok is False
    assert "6-digit" in result.error


def test_tool_reports_validation_failure_without_raising(tmp_path):
    async def run():
        return await GetQuoteTool(make_access([Adapter()], tmp_path)).run()

    result = asyncio.run(run())

    assert result.ok is False
    assert "校验失败" in result.error


def test_kline_states_the_actual_window_it_covered(tmp_path):
    async def run():
        return await GetKlineTool(make_access([Adapter()], tmp_path)).run(
            symbol="600519", period="day", adjust=None, years=1
        )

    result = asyncio.run(run())

    assert result.ok is True
    assert "数据区间" in result.content


def test_kline_flags_a_window_much_shorter_than_requested(tmp_path):
    """The Adapter history spans ~1 month; asking for a year must not pass silently."""
    async def run():
        return await GetKlineTool(make_access([Adapter()], tmp_path)).run(
            symbol="600519", period="day", adjust=None, years=1
        )

    result = asyncio.run(run())

    assert result.ok is True
    # A recent listing asked for a year must be told it only has weeks.
    assert "明显短于请求的 1 年" in result.content
    assert "近 1 年" in result.content


def test_kline_full_history_has_no_short_window_warning(tmp_path):
    class LongAdapter(DataAdapter):
        name = "long"

        def fetch_kline(self, symbol, period, adjust, years):
            dates = pd.date_range("2023-01-01", periods=800, freq="D")
            df = pd.DataFrame({"date": dates, "close": [100.0] * 800})
            return FetchResult(df=df, interface="long_kline")

    async def run():
        return await GetKlineTool(make_access([LongAdapter()], tmp_path)).run(
            symbol="600519", period="day", adjust=None, years=1
        )

    result = asyncio.run(run())

    assert result.ok is True
    assert "数据区间" in result.content
    assert "明显短于" not in result.content


class FailingAdapter(DataAdapter):
    name = "boom"

    def fetch_quote(self, symbol):
        raise AdapterError("upstream exploded")


def test_tool_surfaces_adapter_failure_as_structured_error(tmp_path):
    async def run():
        return await GetQuoteTool(make_access([FailingAdapter()], tmp_path)).run(symbol="600519")

    result = asyncio.run(run())

    assert result.ok is False
    assert "upstream exploded" in result.error


class IndicatorsAdapter(DataAdapter):
    name = "ind"

    def fetch_indicators(self, symbol, years, fields):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-30", "2025-12-31"]),
                "净资产收益率(%)": [32.53, 34.19],
                "资产负债率(%)": [15.19, 16.42],
            }
        )
        return FetchResult(df=df, interface="ind_api")


def test_indicators_tool_flags_a_field_the_source_does_not_have(tmp_path):
    """An unmatched field is a gap, not an empty cell; it must be stated."""
    from finharness.tools.fin.indicators import GetIndicatorsTool

    async def run():
        return await GetIndicatorsTool(make_access([IndicatorsAdapter()], tmp_path)).run(
            symbol="600519", years=3, fields=["ROE", "权益乘数"]
        )

    result = asyncio.run(run())

    assert result.ok is True
    # 权益乘数 has no column in this frame and is named as unavailable.
    assert "权益乘数" in result.content
    assert "无对应列" in result.content
    # ROE resolved, so it is not reported as missing.
    assert "净资产收益率" in result.content
