"""Synchronous adapter protocol isolated behind an async facade."""

from abc import ABC
from typing import Any


class DataAdapter(ABC):
    name = "adapter"

    def fetch_quote(self, symbol: str):
        raise NotImplementedError

    def fetch_kline(self, symbol: str, period: str, adjust: str | None, years: int):
        raise NotImplementedError

    def fetch_indicators(self, symbol: str, years: int, fields: list[str] | None):
        raise NotImplementedError
