"""外部数据源的防腐层（anti-corruption layer）契约（docs 03.5.2）。"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import pandas as pd


class AdapterError(RuntimeError):
    """数据源失败，携带足够细节供回退编排使用。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable

    @property
    def message(self) -> str:
        return str(self)


@dataclass(slots=True)
class FetchResult:
    """数据源载荷，以及实际提供该数据的接口（用于引用）。"""

    df: pd.DataFrame
    interface: str


class DataAdapter(ABC):
    """每个数据源实现其能够提供的语义化抓取方法。

    未实现的方法由基类抛出 ``NotImplementedError``，从而让编排器把
    "该数据源没有此接口" 当作普通的回退原因，而不是程序崩溃。
    """

    name = "adapter"

    def fetch_quote(self, symbol: str) -> FetchResult:
        raise NotImplementedError

    def fetch_kline(self, symbol: str, period: str, adjust: str | None, years: int) -> FetchResult:
        raise NotImplementedError

    def fetch_indicators(self, symbol: str, years: int, fields: list[str] | None) -> FetchResult:
        raise NotImplementedError

    def fetch_financials(self, symbol: str, statement: str, years: int) -> FetchResult:
        raise NotImplementedError

    def fetch_valuation(self, symbol: str, lookback_years: int, indicator: str) -> FetchResult:
        raise NotImplementedError

    def fetch_peers(self, industry: str, fields: list[str] | None) -> FetchResult:
        raise NotImplementedError

    def fetch_news(self, symbol: str | None, topic: str | None, top_n: int) -> FetchResult:
        raise NotImplementedError

    def fetch_announcements(self, symbol: str, since: str, top_n: int) -> FetchResult:
        raise NotImplementedError

    def fetch_research_reports(
        self,
        report_type: str,
        industry: str | None,
        institution: str | None,
        keyword: str | None,
        start_date: str | None,
        end_date: str | None,
        top_n: int,
        with_text: bool,
    ) -> FetchResult:
        raise NotImplementedError

    def fetch_macro(self, indicators: list[str], years: int) -> FetchResult:
        raise NotImplementedError

    def fetch_industry_perf(self, industry: str | None, years: int) -> FetchResult:
        raise NotImplementedError

    def fetch_industry_constituents(self, industry: str) -> FetchResult:
        raise NotImplementedError

    def fetch_index_constituents(self, index: str) -> FetchResult:
        raise NotImplementedError

    # 联网访问并非每个数据源都具备的能力；在此声明该方法，可以让不具备此能力
    # 的数据源通过正常的回退通道报告 "不支持"，而不是抛出看起来像 bug 的
    # AttributeError。
    def fetch_web_search(
        self, query: str, top_n: int, topic: str | None, time_range: str | None
    ) -> FetchResult:
        raise NotImplementedError
