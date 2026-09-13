import pytest

from finharness.data.mapping import (
    DEFAULT_VALUATION_INDICATOR,
    EXCHANGE_PREFIXES,
    UnknownExchangePrefix,
    exchange_prefix,
    indicator_field_matches,
    normalize_valuation_indicator,
    prefixed_symbol,
    select_indicator_columns,
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


def test_blank_valuation_indicator_defaults_to_pe_ttm():
    assert normalize_valuation_indicator(None) == DEFAULT_VALUATION_INDICATOR
    assert normalize_valuation_indicator("  ") == DEFAULT_VALUATION_INDICATOR


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("市盈率", "市盈率(TTM)"),
        ("pe", "市盈率(TTM)"),
        ("PE_TTM", "市盈率(TTM)"),
        ("市净率", "市净率"),
        ("总市值", "总市值"),
        ("市值", "总市值"),
        ("市盈率(静)", "市盈率(静)"),
    ],
)
def test_valuation_indicator_aliases_normalize(alias, expected):
    assert normalize_valuation_indicator(alias) == expected


def test_unknown_valuation_indicator_fails_loudly():
    """Forwarding an unknown metric would return another metric's series."""
    with pytest.raises(ValueError, match="不支持的估值指标"):
        normalize_valuation_indicator("净资产收益率")


@pytest.mark.parametrize(
    ("column", "field", "expected"),
    [
        ("净资产收益率(%)", "ROE", True),   # English shorthand for a Chinese label
        ("总资产报酬率(%)", "ROA", True),
        ("销售毛利率(%)", "毛利率", True),
        ("销售毛利率(%)", "净利润", False),
        ("资产负债率(%)", "负债", True),     # plain substring
    ],
)
def test_indicator_field_matching_resolves_aliases(column, field, expected):
    assert indicator_field_matches(column, field) is expected


def test_select_indicator_columns_keeps_date_and_reports_unmatched():
    columns = ["date", "销售净利率(%)", "净资产收益率(%)", "资产负债率(%)"]

    kept, unmatched = select_indicator_columns(columns, ["ROE", "权益乘数"])

    # ROE resolves via alias; 权益乘数 has no column here and is reported.
    assert kept == ["date", "净资产收益率(%)"]
    assert unmatched == ["权益乘数"]


def test_select_indicator_columns_falls_back_to_all_when_nothing_matches():
    """No match must return a usable frame, not a date-only stub."""
    columns = ["date", "销售净利率(%)", "资产负债率(%)"]

    kept, unmatched = select_indicator_columns(columns, ["毛利率"])

    assert kept == columns
    assert unmatched == ["毛利率"]
