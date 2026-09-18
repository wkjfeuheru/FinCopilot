"""run_backtest：单序列与横截面行为、纪律字段。

该 tool 是量化因子场景的核心，因此这些测试锁定让 backtest 保持诚实的要点：
基准始终运行、样本外优先、试验上限被强制执行，且成本输入确实进入结果。
"""

import asyncio

import numpy as np
import pandas as pd

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.tools.fin.backtest import MAX_POOL_SIZE, MAX_TRIALS, RunBacktestTool


def _prices(symbol: str, n: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    rng = np.random.default_rng(abs(hash(symbol)) % 10_000)
    close = 100 * np.cumprod(1 + rng.normal(0.0006, 0.018, n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        }
    )


class Adapter(DataAdapter):
    name = "fake"

    def fetch_kline(self, symbol, period, adjust, years):
        return FetchResult(df=_prices(symbol), interface="fake_kline")

    def fetch_index_constituents(self, index):
        symbols = ["600000", "600036", "000001", "000002", "600519", "000858"]
        return FetchResult(
            df=pd.DataFrame({"symbol": symbols, "name": symbols}), interface="fake_cons"
        )


def make_access(tmp_path) -> DataAccess:
    settings = Settings(data={"cache_dir": tmp_path / "cache"}, paths={"output_dir": tmp_path / "out"})
    data = DataAccess([Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)
    data.settings = settings
    return data


def run(tool, **kwargs):
    return asyncio.run(tool.run(**kwargs))


# --- 单序列 -----------------------------------------------------------

def test_ma_strategy_reports_sample_out_first_with_a_benchmark(tmp_path):
    result = run(RunBacktestTool(make_access(tmp_path)), symbol="600519", strategy="ma",
                 params={"fast": 20, "slow": 60}, years=2, cost_bps=5)

    assert result.ok is True
    content = result.content
    # 样本外必须在样本内之前报告（阅读顺序保护读者）。
    assert content.index("**样本外**") < content.index("**样本内**")
    assert "买入持有" in content
    assert "样本外相对基准年化超额" in content
    assert "试验披露" in content


def test_factor_expression_drives_a_single_series_backtest(tmp_path):
    result = run(
        RunBacktestTool(make_access(tmp_path)),
        symbol="600519",
        factor_expr="ts_mean(close,20)/ts_mean(close,60)-1",
        years=2,
    )

    assert result.ok is True
    assert "因子表达式" in result.content


def test_momentum_strategy_runs(tmp_path):
    result = run(RunBacktestTool(make_access(tmp_path)), symbol="600519", strategy="momentum",
                 params={"window": 20, "holding": 5}, years=2)

    assert result.ok is True


def test_result_carries_a_nav_frame_for_charting(tmp_path):
    result = run(RunBacktestTool(make_access(tmp_path)), symbol="600519", strategy="ma", years=2)

    frame = result.sources[0].df
    assert {"date", "nav", "benchmark_nav"} <= set(frame.columns)


def test_nav_frame_is_persisted_so_it_can_be_charted(tmp_path):
    """backtest 的 frame 会获得一个 parquet 路径，这正是 make_chart 的 cids
    路径所读取的——没有它，文档化的图表流程就找不到该 frame。"""
    result = run(RunBacktestTool(make_access(tmp_path)), symbol="600519", strategy="ma", years=2)

    path = result.sources[0].parquet_path
    assert path is not None
    loaded = pd.read_parquet(path)
    assert {"date", "nav", "benchmark_nav"} <= set(loaded.columns)


def test_losing_strategy_is_stated_as_ineffective_not_hidden(tmp_path):
    # 恒定价格序列会让每个策略都走平；基准也走平，
    # 因此诚实的结论路径（"未跑赢"）必须可到达而不是崩溃。
    result = run(RunBacktestTool(make_access(tmp_path)), symbol="600519",
                 factor_expr="ts_mean(close,20)-ts_mean(close,20)", years=2)

    assert result.ok is True
    assert "结论" in result.content


# --- 纪律守卫 -------------------------------------------------------

def test_trial_cap_is_enforced(tmp_path):
    result = run(RunBacktestTool(make_access(tmp_path)), symbol="600519",
                 strategy="ma", trials=MAX_TRIALS + 1)

    assert result.ok is False
    assert "试验" in result.error


def test_missing_inputs_are_refused(tmp_path):
    tool = RunBacktestTool(make_access(tmp_path))

    assert run(tool, symbol="600519").ok is False          # 无 strategy/factor
    assert run(tool, strategy="ma").ok is False            # 无 symbol/pool
    assert run(tool, symbol="600519", pool="000300", strategy="ma").ok is False


def test_bad_is_ratio_is_refused(tmp_path):
    result = run(RunBacktestTool(make_access(tmp_path)), symbol="600519", strategy="ma", is_ratio=0.05)

    assert result.ok is False


def test_ma_requires_fast_below_slow(tmp_path):
    result = run(RunBacktestTool(make_access(tmp_path)), symbol="600519", strategy="ma",
                 params={"fast": 60, "slow": 20})

    assert result.ok is False


def test_fees_change_the_result(tmp_path):
    free = run(RunBacktestTool(make_access(tmp_path)), symbol="600519", strategy="ma",
               params={"fast": 10, "slow": 30}, years=2, cost_bps=0)
    costly = run(RunBacktestTool(make_access(tmp_path)), symbol="600519", strategy="ma",
                 params={"fast": 10, "slow": 30}, years=2, cost_bps=50)

    assert free.ok and costly.ok
    assert free.content != costly.content


# --- 横截面 -----------------------------------------------------------

def test_cross_section_produces_ic_groups_and_long_short(tmp_path):
    result = run(
        RunBacktestTool(make_access(tmp_path)),
        pool="000300",
        factor_expr="rank(ts_std(close/ts_delay(close,1)-1,20))",
        params={"groups": 3, "rebalance": 5},
        years=2,
    )

    assert result.ok is True
    content = result.content
    assert "IC" in content
    assert "分组净值" in content
    assert "多空" in content
    assert "生存者偏差" in content  # 已说明前视限制


def test_cross_section_requires_a_cross_sectional_factor(tmp_path):
    result = run(
        RunBacktestTool(make_access(tmp_path)),
        pool="000300",
        factor_expr="ts_mean(close,20)",
        years=2,
    )

    assert result.ok is False
    assert "横截面" in result.error


def test_explicit_code_list_is_accepted_as_a_pool(tmp_path):
    result = run(
        RunBacktestTool(make_access(tmp_path)),
        pool=["600000", "600036", "000001", "000002", "600519"],
        factor_expr="rank(close)",
        params={"groups": 2},
        years=2,
    )

    assert result.ok is True


def test_pool_size_is_capped():
    assert MAX_POOL_SIZE == 300


# --- 长任务进展上报 ----------------------------------------------------
# 面板取数是横截面回测里最慢的一段。客户端只在收到真实事件时才重置空闲看门狗，
# 而服务端心跳是 SSE 注释帧、喂不到它；因此这段取数必须上报进展，否则会话会在
# 静默中被误判为卡死而掐断（docs 03.12）。

def _collect_progress(tool):
    events: list[dict] = []

    async def report(payload):
        events.append(payload)

    tool.progress = report
    return events


def test_cross_section_reports_pool_and_panel_progress(tmp_path):
    tool = RunBacktestTool(make_access(tmp_path))
    events = _collect_progress(tool)

    result = run(
        tool,
        pool="000300",
        factor_expr="rank(ts_std(close/ts_delay(close,1)-1,20))",
        params={"groups": 3, "rebalance": 5},
        years=2,
    )

    assert result.ok is True
    phases = [event["phase"] for event in events]
    # 池规模先于取数进度发出：用户最快能得到的反馈是"池有多大"。
    assert phases[0] == "pool"
    assert "panel" in phases

    panel = [event for event in events if event["phase"] == "panel"]
    fetched = [event["fetched"] for event in panel]
    assert fetched == sorted(fetched)          # 单调递增
    assert fetched[-1] == panel[-1]["total"]   # 收尾必达总数
    assert all(event["total"] > 0 for event in panel)
    assert all(event["elapsed_s"] >= 0 for event in panel)


def test_progress_interval_keeps_the_silence_bounded(tmp_path):
    """上报节奏必须足够密：冷缓存下每次取数约受 1s 节流约束，
    若只在结束时上报一次，取数期间就会出现长达数分钟的静默，客户端看门狗
    会在 90s 处掐断一个仍在正常推进的会话（docs 03.12）。
    """
    class WideAdapter(DataAdapter):
        name = "wide"

        def fetch_kline(self, symbol, period, adjust, years):
            return FetchResult(df=_prices(symbol), interface="fake_kline")

        def fetch_index_constituents(self, index):
            symbols = [f"{600000 + i:06d}" for i in range(25)]
            return FetchResult(
                df=pd.DataFrame({"symbol": symbols, "name": symbols}), interface="fake_cons"
            )

    data = DataAccess(
        [WideAdapter()],
        cache=LocalCache(tmp_path / "cache"),
        settings=Settings(data={"cache_dir": tmp_path / "cache"}, paths={"output_dir": tmp_path / "out"}),
    )
    tool = RunBacktestTool(data)
    events = _collect_progress(tool)

    result = run(tool, pool="000300", factor_expr="rank(close)",
                 params={"groups": 2, "rebalance": 5}, years=2)

    assert result.ok is True
    panel = [event["fetched"] for event in events if event["phase"] == "panel"]
    # 25 只标的：第 10、20 只各上报一次，收尾的第 25 只再上报一次。
    assert panel == [10, 20, 25]


def test_progress_is_optional_so_direct_calls_still_work(tmp_path):
    # 未注入回调（如单测直接调用）时不得报错，也不能依赖进度通道存在。
    result = run(
        RunBacktestTool(make_access(tmp_path)),
        pool="000300",
        factor_expr="rank(close)",
        params={"groups": 2},
        years=2,
    )

    assert result.ok is True


def test_capped_pool_is_disclosed_in_the_result(tmp_path):
    # 指数成分股多于上限时被截断，结果必须写明——否则用户会把结论误当作
    # 对完整成分股池的检验（中证500 的 500 只被静默截到 300 只是原本的隐患）。
    result = run(
        RunBacktestTool(make_access(tmp_path)),
        pool="000300",
        factor_expr="rank(close)",
        params={"groups": 2, "max_symbols": 3},
        years=2,
    )

    assert result.ok is True
    assert "截断" in result.content
    assert "max_symbols" in result.content


def test_uncapped_pool_carries_no_truncation_notice(tmp_path):
    result = run(
        RunBacktestTool(make_access(tmp_path)),
        pool="000300",
        factor_expr="rank(close)",
        params={"groups": 2},
        years=2,
    )

    assert result.ok is True
    assert "截断" not in result.content
