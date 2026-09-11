import asyncio

import pandas as pd

from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter
from finharness.tools.fin.kline import GetKlineTool
from finharness.tools.fin.quote import GetQuoteTool


class Adapter(DataAdapter):
    name = "fake"

    def fetch_quote(self, symbol):
        return pd.DataFrame([{"symbol": symbol, "price": 100.0}])

    def fetch_kline(self, symbol, period, adjust, years):
        return pd.DataFrame([{"date": "2026-01-01", "close": 100.0}])


def test_quote_tool_renders_adapter_data():
    async def run():
        return await GetQuoteTool(DataAccess([Adapter()])).run(symbol="600519")

    result = asyncio.run(run())

    assert result.ok is True
    assert "100" in result.content


def test_kline_rejects_invalid_symbol():
    async def run():
        return await GetKlineTool(DataAccess([Adapter()])).run(
            symbol="bad", period="day", adjust=None, years=1
        )

    result = asyncio.run(run())

    assert result.ok is False
    assert "6-digit" in result.error
