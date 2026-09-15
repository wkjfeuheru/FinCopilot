"""AkShare adapter：同业对比表必须保持可归属且不被污染。

EM 对比 endpoint 会把公司行与两行汇总行混在一起，并且在使用 ``fields`` 时，
过去会丢掉 代码/简称 —— 留下的 frame 调用方既无法辨认，也无法正确求平均
（文档 03.5.2）。
"""

import time

import pandas as pd

from finharness.data.adapters import akshare_adapter
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.adapters.base import AdapterError
from finharness.data.mapping import PEER_COMPANY, PEER_ROW_TYPE_COLUMN, PEER_STAT


class FakeAk:
    """顶替 akshare 模块；返回一张固定的对比表。"""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def stock_zh_valuation_comparison_em(self, symbol: str) -> pd.DataFrame:
        return self._frame.copy()


def peer_frame() -> pd.DataFrame:
    """两家发行人，外加 EM 原样返回的汇总行。"""
    return pd.DataFrame(
        {
            "排名": ["6.0/24", None, None, "1.0"],
            "代码": ["300272", "行业中值", "行业平均", "688169"],
            "简称": ["开能健康", "行业中值", "行业平均", "石头科技"],
            "市盈率-TTM": [117.9, 26.3, 37.4, 18.5],
            "市净率-MRQ": [2.74, 2.79, 3.36, 2.13],
        }
    )


def make_adapter(monkeypatch, frame: pd.DataFrame) -> AkShareAdapter:
    monkeypatch.setattr(akshare_adapter, "_import_akshare", lambda: FakeAk(frame))
    return AkShareAdapter(throttle_seconds=0)


def test_peer_filter_keeps_identity_columns(monkeypatch):
    adapter = make_adapter(monkeypatch, peer_frame())

    df = adapter.fetch_peers("688169", ["市净率", "市盈率"]).df

    # 即使经过字段过滤，这些行仍保持可归属。
    assert {"代码", "简称"}.issubset(df.columns)
    assert {"市净率-MRQ", "市盈率-TTM"}.issubset(df.columns)


def test_peer_rows_are_tagged_as_company_or_statistic(monkeypatch):
    adapter = make_adapter(monkeypatch, peer_frame())

    df = adapter.fetch_peers("688169", ["市净率"]).df

    assert PEER_ROW_TYPE_COLUMN in df.columns
    assert set(df[PEER_ROW_TYPE_COLUMN]) == {PEER_COMPANY, PEER_STAT}
    assert (df[PEER_ROW_TYPE_COLUMN] == PEER_STAT).sum() == 2


def test_peer_companies_sort_before_aggregates(monkeypatch):
    adapter = make_adapter(monkeypatch, peer_frame())

    df = adapter.fetch_peers("688169", ["市净率"]).df

    # 汇总行排在最后，这样天真的整表求平均会明显出错。
    types = list(df[PEER_ROW_TYPE_COLUMN])
    assert types == [PEER_COMPANY, PEER_COMPANY, PEER_STAT, PEER_STAT]


def test_peer_without_fields_keeps_every_column_and_tags_rows(monkeypatch):
    adapter = make_adapter(monkeypatch, peer_frame())

    df = adapter.fetch_peers("688169", None).df

    assert "市盈率-TTM" in df.columns
    assert PEER_ROW_TYPE_COLUMN in df.columns
    assert df.iloc[0][PEER_ROW_TYPE_COLUMN] == PEER_COMPANY


def test_peer_table_without_aggregate_rows_is_untouched(monkeypatch):
    """没有汇总行就无需打标签；frame 必须原样透传。"""
    plain = pd.DataFrame(
        {
            "代码": ["688169", "603486"],
            "简称": ["石头科技", "科沃斯"],
            "市净率-MRQ": [2.13, 2.96],
        }
    )
    adapter = make_adapter(monkeypatch, plain)

    df = adapter.fetch_peers("688169", ["市净率"]).df

    assert PEER_ROW_TYPE_COLUMN not in df.columns
    assert list(df["代码"]) == ["688169", "603486"]


class FakeValuationAk:
    """记录被请求的 indicator；返回一个只有 date/value 两列的 frame。"""

    def __init__(self) -> None:
        self.indicators: list[str] = []

    def stock_zh_valuation_baidu(self, symbol: str, indicator: str, period: str) -> pd.DataFrame:
        self.indicators.append(indicator)
        return pd.DataFrame(
            {"date": ["2026-09-12", "2026-09-11"], "value": [2707.42, 2735.76]}
        )


def make_valuation_adapter(monkeypatch) -> tuple[AkShareAdapter, FakeValuationAk]:
    fake = FakeValuationAk()
    monkeypatch.setattr(akshare_adapter, "_import_akshare", lambda: fake)
    return AkShareAdapter(throttle_seconds=0), fake


def test_valuation_passes_the_requested_indicator_to_the_source(monkeypatch):
    adapter, fake = make_valuation_adapter(monkeypatch)

    adapter.fetch_valuation("000858", 1, "市盈率(TTM)")

    assert fake.indicators == ["市盈率(TTM)"]


def test_valuation_column_names_the_metric_and_unit(monkeypatch):
    """一个光秃秃的 ``value`` 列，正是市值被误读成市盈率倍数的原因。"""
    adapter, _ = make_valuation_adapter(monkeypatch)

    df = adapter.fetch_valuation("000858", 1, "总市值").df

    assert "总市值(亿元)" in df.columns
    assert "value" not in df.columns
    assert df.iloc[0]["总市值(亿元)"] == 2707.42


class FakeIndicatorAk:
    """财务分析指标数据源：中文标签、一个日期列、没有 ROE 字样。"""

    def stock_financial_analysis_indicator(self, symbol: str, start_year: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "日期": ["2026-06-30", "2025-12-31"],
                "净资产收益率(%)": [32.53, 34.19],
                "销售净利率(%)": [50.75, 52.49],
                "资产负债率(%)": [15.19, 16.42],
            }
        )


def test_indicator_field_filter_resolves_english_alias(monkeypatch):
    """``ROE`` 必须能选中 ``净资产收益率(%)``，而不是悄悄匹配不到任何列。"""
    monkeypatch.setattr(akshare_adapter, "_import_akshare", lambda: FakeIndicatorAk())
    adapter = AkShareAdapter(throttle_seconds=0)

    df = adapter.fetch_indicators("600519", 3, ["ROE"]).df

    assert "净资产收益率(%)" in df.columns
    assert "date" in df.columns
    assert "销售净利率(%)" not in df.columns


# --- quote 候选源预算 ------------------------------------------------

def _em_frame() -> pd.DataFrame:
    """EM 快照的形态，即 ``quote`` 归一化重命名之前的样子。"""
    return pd.DataFrame(
        {
            "代码": ["600519"], "名称": ["贵州茅台"], "最新价": [1275.16],
            "涨跌幅": [1.23], "成交量": [34801.0], "成交额": [4.43e9], "换手率": [0.28],
        }
    )


def _tx_frame() -> pd.DataFrame:
    """腾讯日线序列；最后一行是最新的 close。"""
    return pd.DataFrame({"date": ["2026-09-10", "2026-09-11"], "close": [1269.0, 1275.16]})


def make_quote_adapter(monkeypatch, ak) -> AkShareAdapter:
    monkeypatch.setattr(akshare_adapter, "_import_akshare", lambda: ak)
    return AkShareAdapter(throttle_seconds=0)


def test_quote_prefers_the_rich_snapshot_when_available(monkeypatch):
    class Ak:
        def stock_zh_a_spot_em(self):
            return _em_frame()

        def stock_zh_a_hist_tx(self, **kwargs):  # pragma: no cover - 不得运行
            raise AssertionError("the fallback must not be reached")

        def stock_zh_a_spot(self):  # pragma: no cover - 不得运行
            raise AssertionError("the last resort must not be reached")

    result = make_quote_adapter(monkeypatch, Ak()).fetch_quote("600519")

    assert result.interface == "stock_zh_a_spot_em"
    assert result.df.iloc[0]["close"] == 1275.16


def test_quote_moves_on_when_a_candidate_exceeds_its_deadline(monkeypatch):
    """一个卡住的数据源不得耗尽调用方的全部时间预算。"""
    class Ak:
        def stock_zh_a_spot_em(self):
            time.sleep(3.0)  # 远长于截止时间
            return _em_frame()

        def stock_zh_a_hist_tx(self, **kwargs):
            return _tx_frame()

        def stock_zh_a_spot(self):  # pragma: no cover - 不得运行
            raise AssertionError("the last resort must not be reached")

    monkeypatch.setattr(akshare_adapter, "_QUOTE_CANDIDATE_DEADLINE_S", 0.2)
    started = time.perf_counter()

    result = make_quote_adapter(monkeypatch, Ak()).fetch_quote("600519")

    # 早在慢候选源返回之前，就已经回落到腾讯。
    assert result.interface == "stock_zh_a_hist_tx"
    assert time.perf_counter() - started < 1.0


def test_tencent_quote_receives_a_request_timeout_and_prefixed_symbol(monkeypatch):
    class Ak:
        def __init__(self):
            self.kwargs = None

        def stock_zh_a_spot_em(self):
            raise RuntimeError("em unreachable")

        def stock_zh_a_hist_tx(self, **kwargs):
            self.kwargs = kwargs
            return _tx_frame()

        def stock_zh_a_spot(self):  # pragma: no cover - 不得运行
            raise AssertionError("the last resort must not be reached")

    ak = Ak()
    make_quote_adapter(monkeypatch, ak).fetch_quote("600519")

    assert ak.kwargs["timeout"] == akshare_adapter._QUOTE_REQUEST_TIMEOUT_S
    assert ak.kwargs["symbol"] == "sh600519"


def test_repeatedly_failing_candidate_is_short_circuited(monkeypatch):
    """超过失败阈值后，已失效的数据源会被跳过，而不是再次探测。"""
    class Ak:
        def __init__(self):
            self.em_calls = 0

        def stock_zh_a_spot_em(self):
            self.em_calls += 1
            raise RuntimeError("em unreachable")

        def stock_zh_a_hist_tx(self, **kwargs):
            return _tx_frame()

        def stock_zh_a_spot(self):  # pragma: no cover - 不得运行
            raise AssertionError("the last resort must not be reached")

    ak = Ak()
    adapter = make_quote_adapter(monkeypatch, ak)

    for _ in range(3):
        assert adapter.fetch_quote("600519").interface == "stock_zh_a_hist_tx"

    # 前两次查询会调用，之后在其冷却期内被跳过。
    assert ak.em_calls == akshare_adapter._UNHEALTHY_AFTER_FAILURES


def test_all_unhealthy_candidates_are_retried_rather_than_giving_up(monkeypatch):
    """把所有数据源都拉黑不得演变成永久性中断。"""
    class Ak:
        def __init__(self):
            self.recovered = False

        def stock_zh_a_spot_em(self):
            if self.recovered:
                return _em_frame()
            raise RuntimeError("em unreachable")

        def stock_zh_a_hist_tx(self, **kwargs):
            raise RuntimeError("tx unreachable")

        def stock_zh_a_spot(self):
            raise RuntimeError("sina unreachable")

    ak = Ak()
    adapter = make_quote_adapter(monkeypatch, ak)
    for _ in range(akshare_adapter._UNHEALTHY_AFTER_FAILURES):
        try:
            adapter.fetch_quote("600519")
        except AdapterError:
            pass

    ak.recovered = True

    # 所有候选源都处于冷却期，于是整条链路被重试，EM 胜出。
    assert adapter.fetch_quote("600519").interface == "stock_zh_a_spot_em"
