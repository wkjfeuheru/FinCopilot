"""编排测试：cache 复用、adapter 回退与来源追溯。"""

import asyncio
from dataclasses import dataclass

import pandas as pd

from finharness.config.settings import DataSettings, Settings
from finharness.data.access import DataAccess, DataUnavailableError
from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult


def _frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=rows), "close": range(rows)}
    )


@dataclass
class CountingAdapter(DataAdapter):
    name: str = "primary"
    calls: int = 0
    fail: bool = False

    def fetch_kline(self, symbol, period, adjust, years):
        self.calls += 1
        if self.fail:
            raise AdapterError(f"{self.name} down")
        return FetchResult(df=_frame(), interface=f"{self.name}_kline_api")


def make_settings(tmp_path, *, order=("primary", "backup")) -> Settings:
    return Settings(
        data=DataSettings(adapter_order=order, cache_dir=tmp_path / "cache")
    )


def test_endpoint_names_the_interface_that_served_the_data(tmp_path):
    primary = CountingAdapter(name="primary")
    access = DataAccess([primary], settings=make_settings(tmp_path))

    raw = asyncio.run(access.kline("600519"))

    assert raw.endpoint == "primary:primary_kline_api"
    assert raw.from_cache is False


def test_second_read_is_served_from_cache_without_calling_the_adapter(tmp_path):
    primary = CountingAdapter(name="primary")
    access = DataAccess([primary], settings=make_settings(tmp_path))

    first = asyncio.run(access.kline("600519"))
    second = asyncio.run(access.kline("600519"))

    assert primary.calls == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.cache_key == first.cache_key


def test_adapter_order_drives_fallback_and_records_the_working_source(tmp_path):
    primary = CountingAdapter(name="primary", fail=True)
    backup = CountingAdapter(name="backup")
    access = DataAccess(
        [primary, backup], settings=make_settings(tmp_path, order=("primary", "backup"))
    )

    raw = asyncio.run(access.kline("600519"))

    assert primary.calls == 1
    assert backup.calls == 1
    # 这种降级对调用方是静默的，但在来源追溯中可见。
    assert raw.endpoint == "backup:backup_kline_api"


def test_all_sources_failing_raises_data_unavailable_with_reasons(tmp_path):
    primary = CountingAdapter(name="primary", fail=True)
    backup = CountingAdapter(name="backup", fail=True)
    access = DataAccess([primary, backup], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.kline("600519"))
    except DataUnavailableError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")

    assert "primary down" in message
    assert "backup down" in message


def test_invalid_symbol_rejected_before_any_adapter_call(tmp_path):
    primary = CountingAdapter(name="primary")
    access = DataAccess([primary], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.kline("bad"))
    except ValueError as exc:
        assert "6-digit" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

    assert primary.calls == 0


def test_adapter_without_the_endpoint_is_skipped(tmp_path):
    """未实现 fetch_indicators 的数据源不得中断整条链路。"""
    primary = CountingAdapter(name="primary")  # 没有 fetch_indicators
    access = DataAccess([primary], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.indicators("600519"))
    except DataUnavailableError as exc:
        assert "不支持 indicators" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")


def test_quote_cache_is_scoped_per_symbol(tmp_path):
    """某个 symbol 的 quote 绝不能提供给另一个 symbol。

    快照类数据源会抓取整个市场，但 adapter 只保留匹配的那一行；如果 cache key
    不包含 symbol，就会把第一个 symbol 的行发给之后每一次查找。
    """
    class QuoteAdapter(DataAdapter):
        name = "primary"
        calls = 0

        def fetch_quote(self, symbol):
            type(self).calls += 1
            return FetchResult(
                df=pd.DataFrame([{"symbol": symbol, "close": 100.0}]),
                interface="quotes_snapshot",
            )

    access = DataAccess([QuoteAdapter()], settings=make_settings(tmp_path))

    first = asyncio.run(access.quote("600519"))
    second = asyncio.run(access.quote("000858"))
    again = asyncio.run(access.quote("600519"))

    # 每个 symbol 拿到自己的行，而不是第一个 symbol 缓存的行。
    assert first.df.iloc[0]["symbol"] == "600519"
    assert second.df.iloc[0]["symbol"] == "000858"
    # 两次抓取（每个 symbol 各一次）；重复的 symbol 命中 cache。
    assert QuoteAdapter.calls == 2
    assert again.from_cache is True
    assert again.df.iloc[0]["symbol"] == "600519"


def test_valuation_cache_key_includes_the_indicator(tmp_path):
    class ValuationAdapter(DataAdapter):
        name = "primary"
        calls = 0

        def fetch_valuation(self, symbol, lookback_years, indicator):
            type(self).calls += 1
            return FetchResult(df=_frame(), interface="valuation_api")

    access = DataAccess([ValuationAdapter()], settings=make_settings(tmp_path))

    asyncio.run(access.valuation("600519", 1, "市盈率(TTM)"))
    asyncio.run(access.valuation("600519", 1, "pe"))  # 同一指标的别名
    assert ValuationAdapter.calls == 1  # 别名复用规范化后的槽位

    asyncio.run(access.valuation("600519", 1, "市净率"))
    assert ValuationAdapter.calls == 2  # 不同指标对应不同槽位


def test_unknown_valuation_indicator_rejected_before_any_adapter_call(tmp_path):
    class ValuationAdapter(DataAdapter):
        name = "primary"
        calls = 0

        def fetch_valuation(self, symbol, lookback_years, indicator):
            type(self).calls += 1
            return FetchResult(df=_frame(), interface="valuation_api")

    adapter = ValuationAdapter()
    access = DataAccess([adapter], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.valuation("600519", 1, "净资产收益率"))
    except ValueError as exc:
        assert "不支持的估值指标" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

    assert adapter.calls == 0


# --- 瞬时故障重试 -------------------------------------------------

class FlakyAdapter(DataAdapter):
    """失败固定次数后即正常提供数据。"""

    name = "flaky"

    def __init__(self, *, fail_times: int, retryable: bool = True) -> None:
        self.fail_times = fail_times
        self.retryable = retryable
        self.calls = 0

    def fetch_web_search(self, query, top_n, topic=None, time_range=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise AdapterError("联网检索请求失败：瞬断", retryable=self.retryable)
        return FetchResult(df=_frame(), interface="search")


def _no_backoff(monkeypatch):
    monkeypatch.setattr("finharness.data.access._ADAPTER_RETRY_BACKOFF_S", 0.0)


def test_retryable_failure_is_retried_until_it_succeeds(tmp_path, monkeypatch):
    """web 类只有单一数据源，因此一次瞬时抖动不得让调用失败。"""
    _no_backoff(monkeypatch)
    adapter = FlakyAdapter(fail_times=2)
    access = DataAccess([adapter], settings=make_settings(tmp_path))

    raw = asyncio.run(access.web_search("白酒 政策", top_n=5))

    assert raw.endpoint == "flaky:search"
    assert adapter.calls == 3


def test_non_retryable_failure_is_not_retried(tmp_path, monkeypatch):
    """认证/配额类错误在第二次尝试时也会同样失败。"""
    _no_backoff(monkeypatch)
    adapter = FlakyAdapter(fail_times=99, retryable=False)
    access = DataAccess([adapter], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.web_search("白酒 政策", top_n=5))
    except DataUnavailableError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")

    assert adapter.calls == 1


def test_retry_gives_up_after_the_attempt_budget(tmp_path, monkeypatch):
    _no_backoff(monkeypatch)
    adapter = FlakyAdapter(fail_times=99, retryable=True)
    access = DataAccess([adapter], settings=make_settings(tmp_path))

    try:
        asyncio.run(access.web_search("白酒 政策", top_n=5))
    except DataUnavailableError as exc:
        assert "瞬断" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")

    assert adapter.calls == 3


def test_source_without_the_web_method_is_skipped_not_fatal(tmp_path):
    """排在 web adapter 之前的非 web 数据源不得中断请求。

    真实链路：AkShareAdapter 在前，TavilyAdapter 在后。前者没有
    ``fetch_web_search``；它必须被报告为不支持并被跳过。
    """
    class NonWeb(DataAdapter):
        name = "nonweb"

    class WebOnly(DataAdapter):
        name = "web"

        def fetch_web_search(self, query, top_n, topic, time_range):
            return FetchResult(df=_frame(), interface="search")

    access = DataAccess([NonWeb(), WebOnly()], settings=make_settings(tmp_path, order=("nonweb", "web")))

    raw = asyncio.run(access.web_search("白酒 政策", top_n=5))

    assert raw.endpoint == "web:search"


def test_a_report_source_without_the_endpoint_is_skipped(tmp_path):
    """研报 adapter 与 A 股数据源处于同一条链路。

    AkShare 没有 ``fetch_research_reports``，因此它必须报告"不支持"
    并跳过，而不是中断请求。
    """
    from finharness.data.adapters.base import DataAdapter as _DataAdapter

    class NonReport(_DataAdapter):
        name = "akshare"

    class Reports(_DataAdapter):
        name = "eastmoney_report"

        def fetch_research_reports(self, *args, **kwargs):
            return FetchResult(
                df=pd.DataFrame([{"title": "一篇研报", "content": ""}]),
                interface="reports",
            )

    access = DataAccess(
        [NonReport(), Reports()],
        settings=make_settings(tmp_path, order=("akshare", "eastmoney_report")),
    )

    raw = asyncio.run(access.research_reports(top_n=1))

    assert raw.endpoint == "eastmoney_report:reports"
    assert raw.df.iloc[0]["title"] == "一篇研报"


def test_report_cache_key_separates_parameter_sets(tmp_path):
    """不同的过滤器不得共用同一个 cache 槽位。"""
    from finharness.data.adapters.base import DataAdapter as _DataAdapter

    calls = {"n": 0}

    class Reports(_DataAdapter):
        name = "eastmoney_report"

        def fetch_research_reports(self, *args, **kwargs):
            calls["n"] += 1
            return FetchResult(
                df=pd.DataFrame([{"title": f"r{calls['n']}", "content": ""}]),
                interface="reports",
            )

    access = DataAccess([Reports()], settings=make_settings(tmp_path, order=("eastmoney_report",)))

    async def both():
        first = await access.research_reports(industry="证券Ⅱ", top_n=1)
        second = await access.research_reports(industry="养殖业", top_n=1)
        return first, second

    first, second = asyncio.run(both())

    assert calls["n"] == 2
    assert first.df.iloc[0]["title"] != second.df.iloc[0]["title"]


# --- 通用数据集派发（docs 03.5） ---------------------------------------

class DatasetAdapter(DataAdapter):
    """一个能提供数据集目录与通用取数的源；记录每次调用的参数。"""

    name = "fuyao"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.catalogs: list[str] = []

    def list_datasets(self, service):
        self.catalogs.append(service)
        return [{"name": "get_a_share_prices_snapshot", "inputSchema": {"type": "object"}}]

    def fetch_dataset(self, service, dataset, params):
        self.calls.append((service, dataset, dict(params)))
        return FetchResult(
            df=pd.DataFrame({"ticker": ["600519"], "limit": [params.get("limit")]}),
            interface=dataset,
        )


def test_dataset_endpoint_names_the_dataset_that_served_it(tmp_path):
    adapter = DatasetAdapter()
    access = DataAccess([adapter], settings=make_settings(tmp_path, order=("fuyao",)))

    raw = asyncio.run(access.query_dataset("a-share", "get_a_share_prices_snapshot", {}))

    assert raw.endpoint == "fuyao:get_a_share_prices_snapshot"


def test_dataset_cache_is_scoped_per_dataset_and_parameters(tmp_path):
    """不同数据集、不同参数必须各占一个槽位。

    否则一个 limit=5 的结果会被交给 limit=500 的下一次查询——调用方拿到的是
    一份与它所请求的形状不符的数据，且没有任何信号。
    """
    adapter = DatasetAdapter()
    access = DataAccess([adapter], settings=make_settings(tmp_path, order=("fuyao",)))

    first = asyncio.run(access.query_dataset("a-share", "ds_a", {"limit": 5}))
    again = asyncio.run(access.query_dataset("a-share", "ds_a", {"limit": 5}))
    other_params = asyncio.run(access.query_dataset("a-share", "ds_a", {"limit": 9}))
    other_dataset = asyncio.run(access.query_dataset("a-share", "ds_b", {"limit": 5}))

    assert first.from_cache is False
    assert again.from_cache is True
    assert other_params.from_cache is False
    assert other_dataset.from_cache is False
    assert len(adapter.calls) == 3


def test_dataset_cache_is_scoped_per_service(tmp_path):
    """同名数据集跨服务不共享槽位：服务是寻址的一部分。"""
    adapter = DatasetAdapter()
    access = DataAccess([adapter], settings=make_settings(tmp_path, order=("fuyao",)))

    asyncio.run(access.query_dataset("a-share", "prices_snapshot", {}))
    second = asyncio.run(access.query_dataset("fund", "prices_snapshot", {}))

    assert second.from_cache is False


def test_dataset_catalog_is_not_written_to_the_table_cache(tmp_path):
    """目录是元数据：写进 parquet 会让"上游新增了端点"在 TTL 内不可见，
    而目录正是用来发现这些端点的。"""
    adapter = DatasetAdapter()
    access = DataAccess([adapter], settings=make_settings(tmp_path, order=("fuyao",)))

    first = asyncio.run(access.dataset_catalog("a-share"))
    second = asyncio.run(access.dataset_catalog("a-share"))

    assert first[0]["name"] == "get_a_share_prices_snapshot"
    # 适配器被问了两次（无本地表缓存），而不是命中缓存。
    assert adapter.catalogs == ["a-share", "a-share"]
    assert second == first


def test_a_source_without_the_dataset_endpoint_is_skipped(tmp_path):
    """排在数据集源之前的数据源必须报告"不支持"并被跳过，而不是中断请求。"""
    class NonDataset(DataAdapter):
        name = "akshare"

    adapter = DatasetAdapter()
    access = DataAccess(
        [NonDataset(), adapter], settings=make_settings(tmp_path, order=("akshare", "fuyao"))
    )

    raw = asyncio.run(access.query_dataset("a-share", "get_a_share_prices_snapshot", {}))

    assert raw.endpoint == "fuyao:get_a_share_prices_snapshot"


def test_dataset_sources_follow_the_configured_adapter_order(tmp_path):
    """同花顺排在 akshare 之前时，请求先到它；这与"行情类数据由谁提供"是同一机制。"""
    class Other(DatasetAdapter):
        name = "other"

    other = Other()
    fuyao = DatasetAdapter()
    access = DataAccess(
        [other, fuyao], settings=make_settings(tmp_path, order=("fuyao", "other"))
    )

    raw = asyncio.run(access.query_dataset("a-share", "ds", {}))

    assert raw.endpoint == "fuyao:ds"
    assert other.calls == []


def test_catalog_failure_when_no_source_can_answer(tmp_path):
    """没有源能给出目录时抛 DataUnavailable，而不是返回一个空目录假装成功。"""
    class Silent(DataAdapter):
        name = "akshare"

    access = DataAccess([Silent()], settings=make_settings(tmp_path, order=("akshare",)))

    try:
        asyncio.run(access.dataset_catalog("a-share"))
    except DataUnavailableError as exc:
        assert "不支持数据集目录" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")


def test_typed_methods_do_not_reach_the_dataset_channel(tmp_path):
    """typed 方法与通用通道各走各的：行情请求不会退化成数据集调用。"""
    adapter = DatasetAdapter()
    access = DataAccess([adapter], settings=make_settings(tmp_path, order=("fuyao",)))

    try:
        asyncio.run(access.quote("600519"))
    except DataUnavailableError as exc:
        assert "不支持 quote" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")

    assert adapter.calls == []


# --- 号段不可识别是回退理由，不是致命错误 --------------------------------

def test_an_unresolvable_symbol_falls_through_to_the_next_source(tmp_path):
    """某源不认识的号段必须让编排继续回退，而不是中断整条请求。

    ``UnknownExchangePrefix`` 继承自 ``ValueError``，因此 ``_fetch`` 里针对它的分支
    必须排在 ``except ValueError: raise`` 之前——否则一个 akshare 本可作答的代码
    （如 ETF）会在第一个适配器上就失败，回退链形同虚设。
    """
    from finharness.data.mapping import UnknownExchangePrefix

    class LimitedSource(DataAdapter):
        name = "fuyao"

        def fetch_quote(self, symbol):
            # 模拟"A 股端点收到 ETF 代码"：该源提供不了这个标的。
            raise UnknownExchangePrefix(f"无法识别交易所前缀：{symbol}")

    class UniversalSource(DataAdapter):
        name = "akshare"
        calls = 0

        def fetch_quote(self, symbol):
            type(self).calls += 1
            return FetchResult(
                df=pd.DataFrame([{"symbol": symbol, "close": 4.58}]),
                interface="spot",
            )

    access = DataAccess(
        [LimitedSource(), UniversalSource()],
        settings=make_settings(tmp_path, order=("fuyao", "akshare")),
    )

    raw = asyncio.run(access.quote("510300"))

    assert raw.endpoint == "akshare:spot"
    assert UniversalSource.calls == 1


def test_an_unresolvable_symbol_reports_every_source_when_all_fail(tmp_path):
    """全部源都不认识时，报错要汇总各源原因，而不是只抛第一个。"""
    from finharness.data.mapping import UnknownExchangePrefix

    class Limited(DataAdapter):
        def __init__(self, name):
            self.name = name

        def fetch_quote(self, symbol):
            raise UnknownExchangePrefix(f"{self.name} 不认识 {symbol}")

    access = DataAccess(
        [Limited("fuyao"), Limited("akshare")],
        settings=make_settings(tmp_path, order=("fuyao", "akshare")),
    )

    try:
        asyncio.run(access.quote("510300"))
    except DataUnavailableError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DataUnavailableError")

    assert "fuyao" in message and "akshare" in message


def test_an_invalid_symbol_still_fails_fast_without_calling_adapters(tmp_path):
    """调用方**格式**非法（非 6 位）仍是致命错误：不该被当成回退理由。

    与"号段不识别"的区别在于：前者无论发给哪个源都非法，后者只是某个源的能力边界。
    """
    from finharness.data.mapping import UnknownExchangePrefix

    class Counting(DataAdapter):
        name = "fuyao"
        calls = 0

        def fetch_quote(self, symbol):
            type(self).calls += 1
            raise UnknownExchangePrefix("nope")

    adapter = Counting()
    access = DataAccess([adapter], settings=make_settings(tmp_path, order=("fuyao",)))

    try:
        asyncio.run(access.quote("abc"))
    except ValueError as exc:
        assert "6-digit" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

    assert adapter.calls == 0
