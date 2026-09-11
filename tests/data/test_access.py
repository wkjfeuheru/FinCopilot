import asyncio

import pandas as pd

from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter


class Adapter(DataAdapter):
    name = "fake"

    def fetch_quote(self, symbol):
        return pd.DataFrame([{"symbol": symbol, "price": 100.0}])


def test_data_access_runs_sync_adapter_off_event_loop():
    raw = asyncio.run(DataAccess([Adapter()]).quote("600519"))

    assert raw.endpoint == "fake:quote"
    assert raw.dataframe.iloc[0]["price"] == 100.0
