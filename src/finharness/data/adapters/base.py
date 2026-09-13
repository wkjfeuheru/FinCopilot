"""Anti-corruption layer contract for external data sources (docs 03.5.2)."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import pandas as pd


class AdapterError(RuntimeError):
    """A source failure carrying enough detail for fallback orchestration."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable

    @property
    def message(self) -> str:
        return str(self)


@dataclass(slots=True)
class FetchResult:
    """A source payload plus the exact interface that served it (for citations)."""

    df: pd.DataFrame
    interface: str


class DataAdapter(ABC):
    """Each source implements the semantic fetches it can serve.

    Unsupported methods raise ``NotImplementedError`` from the base, letting the
    orchestrator treat "this source has no such endpoint" as an ordinary
    fallback reason instead of a crash.
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
