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
