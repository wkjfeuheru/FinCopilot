"""Async data facade: cache lookup, adapter fallback, and provenance.

Tools never touch adapters directly. Every read goes cache-first, falls back
through ``settings.data.adapter_order``, writes successful payloads back, and
returns a ``RawData`` whose ``endpoint`` names the interface that actually
served the data.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pandas as pd

from finharness.config.settings import Settings
from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult
from finharness.data.cache import LocalCache, make_lookup_key
from finharness.data.citation import fingerprint_series
from finharness.data.mapping import normalize_valuation_indicator
from finharness.data.raw import RawData


class DataUnavailableError(RuntimeError):
    """Raised when every configured source fails for a request."""


def validate_symbol(symbol: str) -> str:
    if not re.fullmatch(r"\d{6}", str(symbol)):
        raise ValueError("symbol must be a 6-digit A-share code")
    return symbol


def _data_date(df: pd.DataFrame | None, fallback: str) -> str:
    """Derive the data's own date so TTLs follow the data, not the fetch."""
    if df is None or not len(df):
        return fallback
    for column in ("date", "日期", "公告日期", "报告期", "end_date"):
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce").dropna()
            if len(parsed):
                return parsed.max().date().isoformat()
    period_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
    if period_cols:
        latest = max(period_cols)
        return f"{latest[:4]}-{latest[4:6]}-{latest[6:]}"
    return fallback


class DataAccess:
    """The only data entry point available to tools."""

    def __init__(
        self,
        adapters: list[DataAdapter],
        *,
        cache: LocalCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.adapters = adapters
        self.settings = settings
        self.cache = cache
        if self.cache is None and settings is not None:
            self.cache = LocalCache(settings.data.cache_dir)

    # -- orchestration --------------------------------------------------------
    def _ordered_adapters(self) -> list[DataAdapter]:
        """Apply ``settings.data.adapter_order`` when configured."""
        if self.settings is None:
            return list(self.adapters)
        order = list(self.settings.data.adapter_order)
        by_name = {adapter.name: adapter for adapter in self.adapters}
        ordered = [by_name[name] for name in order if name in by_name]
        ordered.extend(a for a in self.adapters if a.name not in order)
        return ordered

    def _ttl_days(self, kind: str) -> int:
        if self.settings is None:
            return 1
        return int(self.settings.data.cache_ttl_days.get(kind, 1))

    async def _fetch(
        self,
        *,
        kind: str,
        cache_params: dict[str, Any],
        method: str,
        args: tuple[Any, ...],
    ) -> RawData:
        """Cache-first, then adapter fallback in configured order."""
        lookup_key = make_lookup_key(kind=kind, params=cache_params)
        if self.cache is not None:
            cached = self.cache.get(lookup_key)
            if cached is not None:
                df, entry = cached
                return self._raw(
                    df=df, endpoint=entry.endpoint, params=cache_params,
                    from_cache=True, cache_key=entry.cache_key,
                    parquet_path=entry.file_path, data_date=entry.data_date,
                )

        errors: list[str] = []
        for adapter in self._ordered_adapters():
            try:
                result = await asyncio.to_thread(getattr(adapter, method), *args)
            except NotImplementedError:
                errors.append(f"{adapter.name}: 不支持 {kind}")
                continue
            except AdapterError as exc:
                errors.append(f"{adapter.name}: {exc.message}")
                continue
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate adapter faults
                errors.append(f"{adapter.name}: {type(exc).__name__}: {exc}")
                continue

            df = result.df if isinstance(result, FetchResult) else result
            if not isinstance(df, pd.DataFrame):
                errors.append(f"{adapter.name}: 返回非表格数据")
                continue
            interface = result.interface if isinstance(result, FetchResult) else kind
            endpoint = f"{adapter.name}:{interface}"
            data_date = _data_date(df, LocalCache.today())
            if self.cache is not None:
                entry = await self.cache.put(
                    lookup_key=lookup_key,
                    endpoint=endpoint,
                    params=cache_params,
                    data_date=data_date,
                    df=df,
                    ttl_days=self._ttl_days(kind),
                )
                if entry is not None:
                    return self._raw(
                        df=df, endpoint=endpoint, params=cache_params,
                        cache_key=entry.cache_key, parquet_path=entry.file_path,
                        data_date=data_date,
                    )
            return self._raw(df=df, endpoint=endpoint, params=cache_params, data_date=data_date)

        raise DataUnavailableError("; ".join(errors) or "no data adapter configured")

    @staticmethod
    def _raw(
        *,
        df: pd.DataFrame | None,
        endpoint: str,
        params: dict[str, Any],
        data_date: str,
        from_cache: bool = False,
        cache_key: str | None = None,
        parquet_path: str | None = None,
    ) -> RawData:
        return RawData(
            kind="df",
            df=df,
            endpoint=endpoint,
            params=params,
            data_date=data_date,
            from_cache=from_cache,
            cache_key=cache_key,
            parquet_path=parquet_path,
        )

    # -- semantic methods -----------------------------------------------------
    async def quote(self, symbol: str) -> RawData:
        """Latest market snapshot for one symbol.

        The cache key is symbol-scoped even though snapshot sources
        (``stock_zh_a_spot_em``) fetch the whole market: the adapter returns only
        the matching row, so a symbol-independent key would serve the first
        symbol's row to every later one. Sharing one market-wide payload is only
        sound once the *unfiltered* snapshot is cached and filtered on read; the
        per-symbol key is what the current adapter contract can honour.
        """
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="quote", cache_params={"symbol": symbol},
            method="fetch_quote", args=(symbol,),
        )

    async def kline(self, symbol: str, period: str = "day", adjust: str | None = None, years: int = 1) -> RawData:
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="kline",
            cache_params={"symbol": symbol, "period": period, "adjust": adjust, "years": years},
            method="fetch_kline", args=(symbol, period, adjust, years),
        )

    async def indicators(self, symbol: str, years: int = 3, fields: list[str] | None = None) -> RawData:
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="indicators",
            cache_params={"symbol": symbol, "years": years, "fields": fields},
            method="fetch_indicators", args=(symbol, years, fields),
        )

    async def financials(self, symbol: str, statement: str = "利润", years: int = 3) -> RawData:
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="financials",
            cache_params={"symbol": symbol, "statement": statement, "years": years},
            method="fetch_financials", args=(symbol, statement, years),
        )

    async def valuation(
        self, symbol: str, lookback_years: int = 1, indicator: str | None = None
    ) -> RawData:
        """One valuation series; ``indicator`` selects which metric.

        Normalising here (not in the adapter) keeps aliases on a single cache
        slot and makes the canonical name part of the lookup key.
        """
        symbol = validate_symbol(symbol)
        canonical = normalize_valuation_indicator(indicator)
        return await self._fetch(
            kind="valuation",
            cache_params={
                "symbol": symbol,
                "lookback_years": lookback_years,
                "indicator": canonical,
            },
            method="fetch_valuation", args=(symbol, lookback_years, canonical),
        )

    async def peers(self, symbol: str, fields: list[str] | None = None) -> RawData:
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="peers",
            cache_params={"symbol": symbol, "fields": fields},
            method="fetch_peers", args=(symbol, fields),
        )

    async def news(self, symbol: str | None = None, topic: str | None = None, top_n: int = 10) -> RawData:
        if symbol is not None:
            symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="news",
            cache_params={"symbol": symbol, "topic": topic, "top_n": top_n},
            method="fetch_news", args=(symbol, topic, top_n),
        )

    async def announcements(self, symbol: str, since: str, top_n: int = 20) -> RawData:
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="announcements",
            cache_params={"symbol": symbol, "since": since, "top_n": top_n},
            method="fetch_announcements", args=(symbol, since, top_n),
        )

    async def web_search(
        self,
        query: str,
        top_n: int = 5,
        topic: str | None = None,
        time_range: str | None = None,
    ) -> RawData:
        """External web search (docs 03.4).

        ``kind="web"`` selects the short web TTL; the ``op`` field keeps search
        and fetch results in separate cache slots, since both share that kind.
        """
        return await self._fetch(
            kind="web",
            cache_params={
                "op": "search",
                "query": query,
                "top_n": top_n,
                "topic": topic,
                "time_range": time_range,
            },
            method="fetch_web_search",
            args=(query, top_n, topic, time_range),
        )

    async def fetch_url(self, url: str, query: str | None = None) -> RawData:
        """Fetch one page's content through the search provider (docs 03.4).

        The request leaves from the provider's servers, not this host, so there
        is no SSRF surface here; ``TavilyAdapter`` only screens the scheme.
        """
        return await self._fetch(
            kind="web",
            cache_params={"op": "extract", "url": url, "query": query},
            method="fetch_url",
            args=(url, query),
        )

    # -- helpers --------------------------------------------------------------
    def fingerprint(self, df: pd.DataFrame | None) -> str:
        if df is None or not len(df):
            return fingerprint_series([])
        return fingerprint_series(df.head(50).to_dict("records"))
