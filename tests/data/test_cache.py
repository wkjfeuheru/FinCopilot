import asyncio

import pandas as pd

from finharness.data.cache import LocalCache, make_cache_key, make_lookup_key


def _frame(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=rows), "close": range(rows)})


def test_lookup_key_is_date_independent_and_param_sensitive():
    a = make_lookup_key(kind="kline", params={"symbol": "600519", "years": 1})
    b = make_lookup_key(kind="kline", params={"years": 1, "symbol": "600519"})  # order-insensitive
    c = make_lookup_key(kind="kline", params={"symbol": "600519", "years": 3})

    assert a == b
    assert a != c


def test_cache_key_changes_with_date_endpoint_and_params():
    base = dict(endpoint="akshare:x", params={"symbol": "600519"}, data_date="2026-01-01")

    assert make_cache_key(**base) != make_cache_key(**{**base, "data_date": "2026-01-02"})
    assert make_cache_key(**base) != make_cache_key(**{**base, "endpoint": "tushare:x"})
    assert make_cache_key(**base) != make_cache_key(
        **{**base, "params": {"symbol": "000001"}}
    )


def test_cache_round_trip_and_hit(tmp_path):
    cache = LocalCache(tmp_path)
    key = make_lookup_key(kind="kline", params={"symbol": "600519"})

    async def run():
        await cache.put(
            lookup_key=key, endpoint="akshare:x", params={"symbol": "600519"},
            data_date="2026-01-01", df=_frame(), ttl_days=1,
        )
        return cache.get(key)

    hit = asyncio.run(run())

    assert hit is not None
    frame, entry = hit
    assert len(frame) == 3
    assert entry.endpoint == "akshare:x"
    assert cache.stats().hits == 1


def test_cache_miss_for_unknown_key(tmp_path):
    cache = LocalCache(tmp_path)

    assert cache.get("does-not-exist") is None
    assert cache.stats().misses == 1


def test_cache_expired_entry_is_a_miss(tmp_path):
    cache = LocalCache(tmp_path)
    key = make_lookup_key(kind="kline", params={})

    async def run():
        entry = await cache.put(
            lookup_key=key, endpoint="a:x", params={}, data_date="2020-01-01",
            df=_frame(), ttl_days=0,
        )
        return entry

    asyncio.run(run())

    assert cache.get(key) is None


def test_cache_empty_frame_is_not_persisted(tmp_path):
    cache = LocalCache(tmp_path)
    key = make_lookup_key(kind="kline", params={})

    async def run():
        return await cache.put(
            lookup_key=key, endpoint="a:x", params={}, data_date="2026-01-01",
            df=pd.DataFrame(), ttl_days=1,
        )

    assert asyncio.run(run()) is None
    assert cache.get(key) is None
    assert cache.stats().entries == 0


def test_cache_write_is_idempotent_for_same_lookup(tmp_path):
    cache = LocalCache(tmp_path)
    key = make_lookup_key(kind="kline", params={"symbol": "600519"})

    async def run():
        for _ in range(2):
            await cache.put(
                lookup_key=key, endpoint="a:x", params={"symbol": "600519"},
                data_date="2026-01-01", df=_frame(5), ttl_days=1,
            )

    asyncio.run(run())

    # INSERT OR REPLACE keeps a single row for the same lookup key.
    assert cache.stats().entries == 1


def test_cache_roundtrip_preserves_column_names(tmp_path):
    cache = LocalCache(tmp_path)
    key = make_lookup_key(kind="quote", params={"scope": "all_market"})

    async def run():
        await cache.put(
            lookup_key=key, endpoint="a:quote", params={"scope": "all_market"},
            data_date="2026-01-01", df=_frame(), ttl_days=1,
        )
        return cache.get(key)

    frame, _ = asyncio.run(run())

    assert list(frame.columns) == ["date", "close"]
