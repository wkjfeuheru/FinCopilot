import pytest

from finharness.data.mapping import (
    EXCHANGE_PREFIXES,
    UnknownExchangePrefix,
    exchange_prefix,
    prefixed_symbol,
)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("600519", "SH"),  # 沪市主板
        ("688981", "SH"),  # 科创板
        ("900901", "SH"),  # 沪市B股
        ("000001", "SZ"),  # 深市主板
        ("002594", "SZ"),  # 中小板并入主板
        ("300750", "SZ"),  # 创业板
        ("200011", "SZ"),  # 深市B股
        ("430047", "BJ"),  # 北交所
        ("831010", "BJ"),
        ("870204", "BJ"),
        ("920002", "BJ"),
    ],
)
def test_exchange_prefix_covers_every_board(symbol, expected):
    assert exchange_prefix(symbol) == expected


def test_prefixed_symbol_upper_and_lower():
    assert prefixed_symbol("600519") == "SH600519"
    assert prefixed_symbol("600519", lower=True) == "sh600519"
    assert prefixed_symbol("300750", lower=True) == "sz300750"


@pytest.mark.parametrize("symbol", ["999999", "123456", "abc123"])
def test_unknown_segment_fails_loudly_instead_of_guessing(symbol):
    """A guessed prefix makes some upstream interfaces return an empty frame,
    which would then be cached as "no data"."""
    with pytest.raises(UnknownExchangePrefix):
        exchange_prefix(symbol)


def test_prefix_table_has_no_single_character_keys():
    assert all(len(key) >= 2 for key in EXCHANGE_PREFIXES)
