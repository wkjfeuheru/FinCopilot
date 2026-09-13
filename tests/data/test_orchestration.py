"""Orchestration tests: cache reuse, adapter fallback, and provenance."""

import asyncio
from dataclasses import dataclass

import pandas as pd

from finharness.config.settings import DataSettings, Settings
from finharness.data.access import DataAccess, DataUnavailableError
from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult
from finharness.data.cache import LocalCache


def _frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=rows), "close": range(rows)}
    )


@dataclass
class CountingAdapter(DataAdapter):
    name: str = "primary"
    calls: int = 0
    fail: bool = False

    def fetch_kline(self, symbol, period, adjust, years):
        self.calls += 1
        if self.fail:
            raise AdapterError(f"{self.name} down")
        return FetchResult(df=_frame(), interface=f"{self.name}_kline_api")


def make_settings(tmp_path, *, order=("primary", "backup")) -> Settings:
    return Settings(
        data=DataSettings(adapter_order=order, cache_dir=tmp_path / "cache")
    )


def test_endpoint_names_the_interface_that_served_the_data(tmp_path):
    primary = CountingAdapter(name="primary")
    access = DataAccess([primary], settings=make_settings(tmp_path))

    raw = asyncio.run(access.kline("600519"))

    assert raw.endpoint == "primary:primary_kline_api"
    assert raw.from_cache is False


def test_second_read_is_served_from_cache_without_calling_the_adapter(tmp_path):
    primary = CountingAdapter(name="primary")
    access = DataAccess([primary], settings=make_settings(tmp_path))

    first = asyncio.run(access.kline("600519"))
    second = asyncio.run(access.kline("600519"))

    assert primary.calls == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.cache_key == first.cache_key


def test_adapter_order_drives_fallback_and_records_the_working_source(tmp_path):
    primary = CountingAdapter(name="primary", fail=True)
    backup = CountingAdapter(name="backup")
    access = DataAccess(
        [primary, backup], settings=make_settings(tmp_path, order=("primary", "backup"))
    )

    raw = asyncio.run(access.kline("600519"))

    assert primary.calls == 1
    assert backup.calls == 1
    # The degradation is silent to the caller but visible in provenance.
    assert raw.endpoint == "backup:backup_kline_api"


def test_all_sources_failing_raises_data_unavailable_with_reasons(tmp_path):
    primary = CountingAdapter(name="primary", fail=True)
    backup = CountingAdapter(name="backup", fail=True)
    access = DataAccess([primary, backup], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.kline("600519"))
    except DataUnavailableError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")

    assert "primary down" in message
    assert "backup down" in message


def test_invalid_symbol_rejected_before_any_adapter_call(tmp_path):
    primary = CountingAdapter(name="primary")
    access = DataAccess([primary], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.kline("bad"))
    except ValueError as exc:
        assert "6-digit" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

    assert primary.calls == 0


def test_adapter_without_the_endpoint_is_skipped(tmp_path):
    """A source that does not implement fetch_indicators must not abort the chain."""
    primary = CountingAdapter(name="primary")  # has no fetch_indicators
    access = DataAccess([primary], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.indicators("600519"))
    except DataUnavailableError as exc:
        assert "不支持 indicators" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")


def test_quote_cache_is_scoped_per_symbol(tmp_path):
    """A symbol's quote must never be served to a different symbol.

    Snapshot sources fetch the whole market, but the adapter keeps only the
    matching row; a symbol-independent cache key would hand the first symbol's
    row to every later lookup.
    """
    class QuoteAdapter(DataAdapter):
        name = "primary"
        calls = 0

        def fetch_quote(self, symbol):
            type(self).calls += 1
            return FetchResult(
                df=pd.DataFrame([{"symbol": symbol, "close": 100.0}]),
                interface="quotes_snapshot",
            )

    access = DataAccess([QuoteAdapter()], settings=make_settings(tmp_path))

    first = asyncio.run(access.quote("600519"))
    second = asyncio.run(access.quote("000858"))
    again = asyncio.run(access.quote("600519"))

    # Each symbol gets its own row, not the first symbol's cached row.
    assert first.df.iloc[0]["symbol"] == "600519"
    assert second.df.iloc[0]["symbol"] == "000858"
    # Two fetches (one per symbol); the repeated symbol is a cache hit.
    assert QuoteAdapter.calls == 2
    assert again.from_cache is True
    assert again.df.iloc[0]["symbol"] == "600519"


def test_valuation_cache_key_includes_the_indicator(tmp_path):
    class ValuationAdapter(DataAdapter):
        name = "primary"
        calls = 0

        def fetch_valuation(self, symbol, lookback_years, indicator):
            type(self).calls += 1
            return FetchResult(df=_frame(), interface="valuation_api")

    access = DataAccess([ValuationAdapter()], settings=make_settings(tmp_path))

    asyncio.run(access.valuation("600519", 1, "市盈率(TTM)"))
    asyncio.run(access.valuation("600519", 1, "pe"))  # alias of the same metric
    assert ValuationAdapter.calls == 1  # alias reuses the canonical slot

    asyncio.run(access.valuation("600519", 1, "市净率"))
    assert ValuationAdapter.calls == 2  # a different metric is a different slot


def test_unknown_valuation_indicator_rejected_before_any_adapter_call(tmp_path):
    class ValuationAdapter(DataAdapter):
        name = "primary"
        calls = 0

        def fetch_valuation(self, symbol, lookback_years, indicator):
            type(self).calls += 1
            return FetchResult(df=_frame(), interface="valuation_api")

    adapter = ValuationAdapter()
    access = DataAccess([adapter], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.valuation("600519", 1, "净资产收益率"))
    except ValueError as exc:
        assert "不支持的估值指标" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

    assert adapter.calls == 0
