"""异步数据门面：缓存查询、适配器回退与来源追溯。

工具从不直接接触适配器。每次读取都先查缓存，再按
``settings.data.adapter_order`` 逐级回退，成功后将数据写回缓存，
并返回一个 ``RawData``，其 ``endpoint`` 标明实际提供该数据的
接口。
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any

import pandas as pd

from finharness.config.settings import Settings
from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult
from finharness.data.cache import LocalCache, make_lookup_key
from finharness.data.citation import fingerprint_series
from finharness.data.mapping import normalize_valuation_indicator
from finharness.data.raw import RawData


class DataUnavailableError(RuntimeError):
    """当某次请求的所有已配置数据源都失败时抛出。"""


# 可重试的数据源故障（TLS 握手重置、429、5xx）通常只是瞬时抖动。
# 在放弃前短暂重试，对 web 类型尤为重要，因为它只有一个适配器：
# 若没有该重试，一次瞬时错误就会导致整个调用失败，
# 尽管下一次尝试本可成功。
_ADAPTER_MAX_ATTEMPTS = 3
_ADAPTER_RETRY_BACKOFF_S = 0.5


def validate_symbol(symbol: str) -> str:
    """校验标的为 6 位 A 股代码，格式不符则抛出 ``ValueError``。"""
    if not re.fullmatch(r"\d{6}", str(symbol)):
        raise ValueError("symbol must be a 6-digit A-share code")
    return symbol


def _now_stamp() -> str:
    """本次取数的抓取时刻，与缓存条目所用的时间戳格式一致（本地时区、秒级）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _data_date(df: pd.DataFrame | None, fallback: str) -> str:
    """推导数据自身的日期，使 TTL 跟随数据而非抓取时刻。"""
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
    """工具可用的唯一数据入口。"""

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

    # -- 编排 --------------------------------------------------------
    def _ordered_adapters(self) -> list[DataAdapter]:
        """在已配置时应用 ``settings.data.adapter_order``。"""
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
        """先查缓存，再按配置顺序回退到适配器。"""
        lookup_key = make_lookup_key(kind=kind, params=cache_params)
        if self.cache is not None:
            cached = self.cache.get(lookup_key)
            if cached is not None:
                df, entry = cached
                return self._raw(
                    df=df, endpoint=entry.endpoint, params=cache_params,
                    from_cache=True, cache_key=entry.cache_key,
                    parquet_path=entry.file_path, data_date=entry.data_date,
                    fetched_at=entry.created_ts,
                )

        errors: list[str] = []
        for adapter in self._ordered_adapters():
            result: Any = None
            for attempt in range(1, _ADAPTER_MAX_ATTEMPTS + 1):
                try:
                    # ``getattr`` 保持在 try 内部：未实现该语义方法的
                    # 适配器必须落到下一个数据源，而不是中止
                    # 整个请求。
                    result = await asyncio.to_thread(getattr(adapter, method), *args)
                    break
                except NotImplementedError:
                    errors.append(f"{adapter.name}: 不支持 {kind}")
                    break
                except AdapterError as exc:
                    # 仅重试瞬时故障；鉴权与配额错误
                    # 在第二次尝试时也会同样失败。
                    if exc.retryable and attempt < _ADAPTER_MAX_ATTEMPTS:
                        await asyncio.sleep(_ADAPTER_RETRY_BACKOFF_S * attempt)
                        continue
                    errors.append(f"{adapter.name}: {exc.message}")
                    break
                except ValueError:
                    raise
                except Exception as exc:  # noqa: BLE001 - 隔离适配器故障
                    errors.append(f"{adapter.name}: {type(exc).__name__}: {exc}")
                    break
            if result is None:
                continue

            df = result.df if isinstance(result, FetchResult) else result
            if not isinstance(df, pd.DataFrame):
                errors.append(f"{adapter.name}: 返回非表格数据")
                continue
            interface = result.interface if isinstance(result, FetchResult) else kind
            endpoint = f"{adapter.name}:{interface}"
            data_date = _data_date(df, LocalCache.today())
            fetched_at = _now_stamp()
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
                        data_date=data_date, fetched_at=fetched_at,
                    )
            return self._raw(
                df=df, endpoint=endpoint, params=cache_params,
                data_date=data_date, fetched_at=fetched_at,
            )

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
        fetched_at: str | None = None,
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
            fetched_at=fetched_at,
        )

    # -- 语义方法 -----------------------------------------------------
    async def quote(self, symbol: str) -> RawData:
        """单个标的的最新行情快照。

        尽管快照数据源（``stock_zh_a_spot_em``）会抓取整个市场，缓存键仍按
        标的作用域划分：适配器只返回匹配的那一行，因此与标的无关的键会把
        第一个标的的行返回给之后的每一个标的。只有当缓存的是*未过滤*
        的快照并在读取时再过滤，共享整份全市场数据才成立；
        按标的划分的键才是当前适配器契约所
        能够保证的。
        """
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="quote", cache_params={"symbol": symbol},
            method="fetch_quote", args=(symbol,),
        )

    async def kline(self, symbol: str, period: str = "day", adjust: str | None = None, years: int = 1) -> RawData:
        """某标的在指定周期/复权方式下的 K 线序列。"""
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="kline",
            cache_params={"symbol": symbol, "period": period, "adjust": adjust, "years": years},
            method="fetch_kline", args=(symbol, period, adjust, years),
        )

    async def indicators(self, symbol: str, years: int = 3, fields: list[str] | None = None) -> RawData:
        """某标的的财务指标序列，可按字段过滤列。"""
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="indicators",
            cache_params={"symbol": symbol, "years": years, "fields": fields},
            method="fetch_indicators", args=(symbol, years, fields),
        )

    async def financials(self, symbol: str, statement: str = "利润", years: int = 3) -> RawData:
        """某标的指定报表（利润/资产/现金流等）的财务数据。"""
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="financials",
            cache_params={"symbol": symbol, "statement": statement, "years": years},
            method="fetch_financials", args=(symbol, statement, years),
        )

    async def valuation(
        self, symbol: str, lookback_years: int = 1, indicator: str | None = None
    ) -> RawData:
        """单条估值序列；``indicator`` 选择具体指标。

        在此处（而非适配器中）做归一化，可让各种别名落在同一个缓存槽，
        并使规范名称成为查找键的一部分。
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
        """某标的的同行估值对比数据。"""
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="peers",
            cache_params={"symbol": symbol, "fields": fields},
            method="fetch_peers", args=(symbol, fields),
        )

    async def news(self, symbol: str | None = None, topic: str | None = None, top_n: int = 10) -> RawData:
        """个股或主题新闻列表。"""
        if symbol is not None:
            symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="news",
            cache_params={"symbol": symbol, "topic": topic, "top_n": top_n},
            method="fetch_news", args=(symbol, topic, top_n),
        )

    async def announcements(self, symbol: str, since: str, top_n: int = 20) -> RawData:
        """某标的在指定起始日期之后的公告列表。"""
        symbol = validate_symbol(symbol)
        return await self._fetch(
            kind="announcements",
            cache_params={"symbol": symbol, "since": since, "top_n": top_n},
            method="fetch_announcements", args=(symbol, since, top_n),
        )

    async def research_reports(
        self,
        report_type: str = "行业",
        industry: str | None = None,
        institution: str | None = None,
        keyword: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: int = 10,
        with_text: bool = False,
    ) -> RawData:
        """东方财富研究报告（docs 03.4）。

        每个参数都是缓存键的一部分，因此不同的行业、
        机构、时间窗口或文本设置都会对应不同的缓存槽，
        而不会命中过期数据。
        """
        return await self._fetch(
            kind="reports",
            cache_params={
                "report_type": report_type,
                "industry": industry,
                "institution": institution,
                "keyword": keyword,
                "start_date": start_date,
                "end_date": end_date,
                "top_n": top_n,
                "with_text": with_text,
            },
            method="fetch_research_reports",
            args=(report_type, industry, institution, keyword, start_date, end_date, top_n, with_text),
        )

    async def web_search(
        self,
        query: str,
        top_n: int = 5,
        topic: str | None = None,
        time_range: str | None = None,
    ) -> RawData:
        """外部网络搜索（docs 03.4）。

        ``kind="web"`` 会选择较短的 web TTL。
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

    async def macro(self, indicators: list[str], years: int = 3) -> RawData:
        """长表形式的宏观经济序列（docs 03.4）。

        指标在生成缓存键前先做规范化，使别名与其 slug 共享同一个缓存槽；
        列表会排序，因此顺序不会导致键被拆分。
        """
        from finharness.data.mapping import normalize_macro_indicator

        canonical = sorted({normalize_macro_indicator(name) for name in indicators})
        return await self._fetch(
            kind="macro",
            cache_params={"indicators": canonical, "years": years},
            method="fetch_macro",
            args=(canonical, years),
        )

    async def industry_perf(self, industry: str | None = None, years: int = 1) -> RawData:
        """申万行业指数历史；当 industry 为 None 时返回一级行业概览。"""
        return await self._fetch(
            kind="industry",
            cache_params={"op": "perf", "industry": industry, "years": years},
            method="fetch_industry_perf",
            args=(industry, years),
        )

    async def industry_constituents(self, industry: str) -> RawData:
        """申万行业的成分股（横截面选股池的一个来源）。"""
        return await self._fetch(
            kind="industry",
            cache_params={"op": "cons", "industry": industry},
            method="fetch_industry_constituents",
            args=(industry,),
        )

    async def index_constituents(self, index: str) -> RawData:
        """中证指数的成分股（主要的横截面选股池）。"""
        from finharness.data.mapping import normalize_index

        canonical = normalize_index(index)
        return await self._fetch(
            kind="industry",
            cache_params={"op": "index_cons", "index": canonical},
            method="fetch_index_constituents",
            args=(canonical,),
        )

    # -- 辅助函数 --------------------------------------------------------------
    def fingerprint(self, df: pd.DataFrame | None) -> str:
        if df is None or not len(df):
            return fingerprint_series([])
        return fingerprint_series(df.head(50).to_dict("records"))
