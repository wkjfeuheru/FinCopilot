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

    # endpoint 标明实际提供数据的 interface（文档 3.5.2）。
    assert raw.endpoint == "fake:fake_quote_api"
    assert raw.df.iloc[0]["close"] == 100.0
    assert raw.from_cache is False


def test_a_live_fetch_stamps_when_the_data_was_retrieved(tmp_path):
    """抓取时刻是「这份数据在本地有多旧」的另一半事实（另一半是数据期）。"""
    from finharness.config.settings import Settings
    from finharness.data.cache import LocalCache

    settings = Settings(data={"cache_dir": tmp_path / "cache"})
    data = DataAccess([Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)

    raw = asyncio.run(data.quote("600519"))

    assert raw.fetched_at
    assert raw.from_cache is False


def test_a_cache_hit_reports_the_write_time_not_the_current_time(tmp_path):
    """缓存命中要报出该条目**写入**的时刻。

    若报成当前时刻，一份上个月写入的陈旧数据会显示为刚刚取回——那正是静默复用旧数据
    最难发现的地方。
    """
    from finharness.config.settings import Settings
    from finharness.data.cache import LocalCache

    settings = Settings(data={"cache_dir": tmp_path / "cache"})
    data = DataAccess([Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)

    first = asyncio.run(data.quote("600519"))
    second = asyncio.run(data.quote("600519"))

    assert second.from_cache is True
    assert second.fetched_at == first.fetched_at
    assert second.data_date == first.data_date
