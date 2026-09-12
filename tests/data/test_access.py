import asyncio

import pandas as pd

from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult


class Adapter(DataAdapter):
    name = "fake"

    def fetch_quote(self, symbol):
        return FetchResult(
            df=pd.DataFrame([{"symbol": symbol, "close": 100.0}]),
            interface="fake_quote_api",
        )


def test_data_access_runs_sync_adapter_off_event_loop():
    raw = asyncio.run(DataAccess([Adapter()]).quote("600519"))

    # endpoint names the interface that actually served the data (docs 3.5.2).
    assert raw.endpoint == "fake:fake_quote_api"
    assert raw.df.iloc[0]["close"] == 100.0
    assert raw.from_cache is False
