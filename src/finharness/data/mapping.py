"""Single source of truth for source-interface names, column mappings and
exchange prefixes.

Adding a data interface should only extend the tables here, never scatter
source-specific knowledge across adapters or tools.
"""

from __future__ import annotations

from typing import Final

# --- Endpoint names -------------------------------------------------------
# Maps (adapter, semantic method) -> ordered candidate interface names. The
# first candidate that succeeds is recorded on RawData.endpoint, so citations
# and cache keys always name the interface that actually served the data.
AKSHARE_ENDPOINTS: Final[dict[str, tuple[str, ...]]] = {
    # Quote prefers the EM snapshot (richest: name, pct_change, turnover), then
    # the Tencent daily series whose last row is the latest close. The Sina
    # snapshot is last resort: it paginates the whole market with un-timed
    # requests and its own docs warn it IP-bans repeated callers, so it is only
    # reached when both richer sources are unavailable.
    "quote": ("stock_zh_a_spot_em", "stock_zh_a_hist_tx", "stock_zh_a_spot"),
    "kline": ("stock_zh_a_hist", "stock_zh_a_daily", "stock_zh_a_hist_tx"),
    "indicators": ("stock_financial_analysis_indicator",),
    "financials": ("stock_financial_abstract",),
    "valuation": ("stock_zh_valuation_baidu",),
    "peers": ("stock_zh_valuation_comparison_em",),
    "news": ("stock_news_em",),
    "announcements": ("stock_individual_notice_report",),
}

TUSHARE_ENDPOINTS: Final[dict[str, tuple[str, ...]]] = {
    "quote": ("daily_basic",),
    "kline": ("daily",),
    "indicators": ("fina_indicator",),
    "financials": ("income",),
}

# --- Valuation indicators -------------------------------------------------
# ``stock_zh_valuation_baidu`` serves several metrics through one endpoint and
# returns a bare ``date``/``value`` frame with no metric name. The request must
# therefore carry the indicator, and its unit must be attached at render time —
# otherwise a market-cap figure reads as a PE multiple.
VALUATION_INDICATORS: Final[tuple[str, ...]] = (
    "总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率",
)
# Accepted spellings for each canonical indicator (lower-cased for lookup).
VALUATION_INDICATOR_ALIASES: Final[dict[str, str]] = {
    "总市值": "总市值", "市值": "总市值", "total_mv": "总市值", "market_cap": "总市值",
    "市盈率(ttm)": "市盈率(TTM)", "市盈率": "市盈率(TTM)", "pe": "市盈率(TTM)",
    "pe_ttm": "市盈率(TTM)", "市盈率ttm": "市盈率(TTM)",
    "市盈率(静)": "市盈率(静)", "静态市盈率": "市盈率(静)",
    "pe_static": "市盈率(静)",
    "市净率": "市净率", "pb": "市净率",
    "市现率": "市现率", "pcf": "市现率",
}
# Baidu reports 总市值 in 亿元; the ratio indicators are multiples (倍).
VALUATION_INDICATOR_UNITS: Final[dict[str, str]] = {
    "总市值": "亿元", "市盈率(TTM)": "倍", "市盈率(静)": "倍",
    "市净率": "倍", "市现率": "倍",
}
DEFAULT_VALUATION_INDICATOR: Final[str] = "市盈率(TTM)"


def normalize_valuation_indicator(indicator: str | None) -> str:
    """Map an alias/blank to a canonical indicator, rejecting unknown ones.

    Fails loudly: an unrecognised indicator forwarded blindly to the source
    would return the wrong metric's series under an unlabelled ``value`` column.
    """
    key = str(indicator or "").strip()
    if not key:
        return DEFAULT_VALUATION_INDICATOR
    if key in VALUATION_INDICATORS:
        return key
    mapped = VALUATION_INDICATOR_ALIASES.get(key) or VALUATION_INDICATOR_ALIASES.get(key.lower())
    if mapped is not None:
        return mapped
    raise ValueError(
        f"不支持的估值指标：{indicator}；可选 {'/'.join(VALUATION_INDICATORS)}"
    )


# --- Indicator field filters ----------------------------------------------
# ``get_indicators(fields=[...])`` filters columns by keyword. Callers (and the
# model) name metrics in English shorthand while sources use Chinese labels, so
# a bare substring test silently matched nothing: asking for ``ROE`` dropped the
# very column that held it (``净资产收益率(%)``) with no signal to the caller.
INDICATOR_FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "roe": ("净资产收益率",),
    "roa": ("总资产报酬率", "总资产净利率"),
    "eps": ("每股收益",),
    "bps": ("每股净资产",),
    "pe": ("市盈率",),
    "pb": ("市净率",),
    "pcf": ("市现率",),
    "npm": ("销售净利率", "净利率"),
    "gpm": ("销售毛利率", "毛利率"),
}


def indicator_field_matches(column: str, field: str) -> bool:
    """Whether ``field`` (itself or an alias) names ``column``."""
    label = str(column).lower()
    key = str(field).strip().lower()
    if not key:
        return False
    if key in label:
        return True
    return any(alias.lower() in label for alias in INDICATOR_FIELD_ALIASES.get(key, ()))


def select_indicator_columns(
    columns: tuple[str, ...] | list[str], fields: list[str]
) -> tuple[list[str], list[str]]:
    """Return ``(kept columns, requested fields that matched nothing)``.

    ``date`` is always kept. When no field matches, ``kept`` is every column so
    the caller still gets a usable frame, and every field is reported as
    unmatched rather than being silently swallowed.
    """
    matched: set[str] = set()
    unmatched: list[str] = []
    for field in fields:
        hits = [c for c in columns if str(c) != "date" and indicator_field_matches(c, field)]
        if hits:
            matched.update(hits)
        else:
            unmatched.append(field)
    if not matched:
        return list(columns), unmatched
    kept = [c for c in columns if str(c) == "date" or c in matched]
    return kept, unmatched

# --- Normalized market columns -------------------------------------------
# Internal contract for quote/kline frames (docs 03.5.2).
MARKET_COLUMNS: Final[tuple[str, ...]] = (
    "date", "open", "high", "low", "close", "volume", "amount",
)

# akshare Chinese column name -> internal snake_case name.
AKSHARE_COLUMN_MAP: Final[dict[str, str]] = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
    "代码": "symbol",
    "名称": "name",
    "最新价": "close",
    "股票代码": "symbol",
}

# Column aliases per semantic interface, applied on top of AKSHARE_COLUMN_MAP.
# Only listed interfaces are normalized; others keep their source semantics.
AKSHARE_INTERFACE_COLUMNS: Final[dict[str, dict[str, str]]] = {
    "quote": {
        "代码": "symbol",
        "名称": "name",
        "最新价": "close",
        "涨跌幅": "pct_change",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    },
    "kline": {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    },
}

# --- Peer comparison -------------------------------------------------------
# Identity columns that must survive a `fields` filter: with 代码/简称 dropped
# the caller cannot tell which row is the target and which rows are aggregates,
# which is how a peer table gets read as if every row were a company.
PEER_IDENTITY_COLUMNS: Final[tuple[str, ...]] = ("排名", "代码", "简称")
# The EM comparison table mixes two aggregate rows into the company list,
# labelled in the 代码/简称 columns. They are tagged with PEER_ROW_TYPE_COLUMN
# so a whole-frame mean is visibly wrong instead of silently polluted.
PEER_STAT_LABELS: Final[tuple[str, ...]] = ("行业中值", "行业平均")
PEER_ROW_TYPE_COLUMN: Final[str] = "行类型"
PEER_COMPANY: Final[str] = "公司"
PEER_STAT: Final[str] = "行业统计"

# --- Exchange prefixes ----------------------------------------------------
# Segment-based (not first-digit) so ChiNext, STAR, B-shares and Beijing are
# all classified correctly. Longest prefix wins.
EXCHANGE_PREFIXES: Final[dict[str, str]] = {
    "600": "SH", "601": "SH", "603": "SH", "605": "SH", "688": "SH", "689": "SH",
    "900": "SH",
    "000": "SZ", "001": "SZ", "002": "SZ", "003": "SZ", "300": "SZ", "301": "SZ",
    "200": "SZ",
    "430": "BJ", "830": "BJ", "831": "BJ", "832": "BJ", "833": "BJ", "834": "BJ",
    "835": "BJ", "836": "BJ", "837": "BJ", "838": "BJ", "839": "BJ",
    "870": "BJ", "871": "BJ", "872": "BJ", "873": "BJ", "874": "BJ",
    "875": "BJ", "876": "BJ", "877": "BJ", "878": "BJ", "879": "BJ",
    "920": "BJ",
}


class UnknownExchangePrefix(ValueError):
    """Raised when a 6-digit code matches no known exchange segment."""


def exchange_prefix(symbol: str) -> str:
    """Return ``SH``/``SZ``/``BJ`` for a 6-digit A-share code.

    Fails loudly instead of guessing: a wrong prefix makes some upstream
    interfaces return an empty frame, which would otherwise be cached as
    "no data".
    """
    for length in (3, 2):
        prefix = EXCHANGE_PREFIXES.get(symbol[:length])
        if prefix is not None:
            return prefix
    raise UnknownExchangePrefix(f"无法识别交易所前缀：{symbol}")


def prefixed_symbol(symbol: str, *, lower: bool = False) -> str:
    """Return an exchange-prefixed code such as ``SH600519`` or ``sh600519``."""
    result = f"{exchange_prefix(symbol)}{symbol}"
    return result.lower() if lower else result
