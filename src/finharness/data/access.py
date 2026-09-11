"""Async data facade with adapter isolation and structured failures."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from finharness.data.adapters.base import DataAdapter


class DataUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class RawData:
    kind: str
    dataframe: Any = None
    text: str | None = None
    paths: list[str] | None = None
    endpoint: str = ""
    params: dict[str, Any] | None = None


def validate_symbol(symbol: str) -> str:
    if not re.fullmatch(r"\d{6}", symbol):
        raise ValueError("symbol must be a 6-digit A-share code")
    return symbol


class DataAccess:
    def __init__(self, adapters: list[DataAdapter]):
        self.adapters = adapters

    async def _fetch(self, method: str, *args) -> RawData:
        errors = []
        for adapter in self.adapters:
            try:
                dataframe = await asyncio.to_thread(getattr(adapter, method), *args)
                return RawData(
                    kind="df",
                    dataframe=dataframe,
                    endpoint=f"{adapter.name}:{method.removeprefix('fetch_')}",
                    params={"args": args},
                )
            except Exception as exc:
                errors.append(f"{adapter.name}: {exc}")
        raise DataUnavailableError("; ".join(errors) or "no data adapter configured")

    async def quote(self, symbol: str) -> RawData:
        return await self._fetch("fetch_quote", validate_symbol(symbol))

    async def kline(self, symbol: str, period: str, adjust: str | None, years: int) -> RawData:
        return await self._fetch("fetch_kline", validate_symbol(symbol), period, adjust, years)

    async def indicators(self, symbol: str, years: int, fields: list[str] | None) -> RawData:
        return await self._fetch("fetch_indicators", validate_symbol(symbol), years, fields)
