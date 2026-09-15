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
