import pandas as pd

from finharness.data.citation import CitationRegistry, fingerprint_frame


def test_register_assigns_sequential_cids_and_records_provenance():
    registry = CitationRegistry()
    df = pd.DataFrame({"close": [1, 2, 3]})

    first = registry.register(
        tool="get_quote", endpoint="akshare:stock_zh_a_spot_em", symbol="600519",
        params={"symbol": "600519"}, rows=len(df), cols=len(df.columns),
        fingerprint=fingerprint_frame(df),
    )
    second = registry.register(
        tool="get_kline", endpoint="akshare:stock_zh_a_daily", symbol="600519",
        params={"symbol": "600519"}, rows=1, cols=1, fingerprint="x",
    )

    assert first.cid == "cit_000001"
    assert second.cid == "cit_000002"
    assert registry.get(first.cid) is first


def test_query_filters_by_symbol_and_tool():
    registry = CitationRegistry()
    registry.register(
        tool="get_quote", endpoint="a:q", symbol="600519", params={},
        rows=1, cols=1, fingerprint="x",
    )
    registry.register(
        tool="get_kline", endpoint="a:k", symbol="000001", params={},
        rows=1, cols=1, fingerprint="y",
    )

    assert len(registry.query(symbol="600519")) == 1
    assert len(registry.query(tool="get_kline")) == 1
    assert len(registry.query()) == 2


def test_resolve_symbols_deduplicates_in_order():
    registry = CitationRegistry()
    for symbol in ("600519", "600519", "000001"):
        registry.register(
            tool="t", endpoint="e", symbol=symbol, params={}, rows=1, cols=1, fingerprint="x"
        )

    assert registry.resolve_symbols() == ["600519", "000001"]


def test_appendix_marks_cache_hits_and_lists_ids():
    registry = CitationRegistry()
    registry.register(
        tool="get_quote", endpoint="akshare:stock_zh_a_spot_em", symbol="600519",
        params={}, rows=5, cols=3, fingerprint="abc", from_cache=True,
    )

    appendix = registry.to_appendix_md()

    assert "cit_000001" in appendix
    assert "缓存" in appendix
    assert "akshare:stock_zh_a_spot_em" in appendix


def test_registry_is_bounded_and_evicts_oldest():
    registry = CitationRegistry(max_entries=3)
    for i in range(5):
        registry.register(
            tool="t", endpoint="e", symbol=str(i), params={}, rows=1, cols=1, fingerprint="x"
        )

    assert len(registry.all()) == 3
    assert registry.get("cit_000001") is None
    assert registry.get("cit_000005") is not None


def test_fingerprint_is_stable_and_content_sensitive():
    a = pd.DataFrame({"close": [1, 2]})
    b = pd.DataFrame({"close": [1, 2]})
    c = pd.DataFrame({"close": [1, 3]})

    assert fingerprint_frame(a) == fingerprint_frame(b)
    assert fingerprint_frame(a) != fingerprint_frame(c)
    assert fingerprint_frame(None) == fingerprint_frame(pd.DataFrame())
