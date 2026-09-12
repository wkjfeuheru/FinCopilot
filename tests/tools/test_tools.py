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
