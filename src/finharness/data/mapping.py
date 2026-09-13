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
    # Quote prefers the EM snapshot (richest), then the Sina snapshot, then the
    # Tencent daily series whose last row is the latest close.
    "quote": ("stock_zh_a_spot_em", "stock_zh_a_spot", "stock_zh_a_hist_tx"),
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
