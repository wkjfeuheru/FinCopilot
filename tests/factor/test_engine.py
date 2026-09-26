"""Factor 表达式引擎：白名单安全、运算符与拒绝策略。

安全断言才是重点：随 prompt 驱动的 agent 一起发布的表达式引擎，
绝不能成为代码执行面。
"""

import numpy as np
import pandas as pd
import pytest

from finharness.factor.engine import (
    MAX_WINDOW,
    FactorEngine,
    FactorError,
    is_cross_sectional,
)


def series(n: int = 300, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"AAA": 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))}, index=dates
    )


def panel(n: int = 300) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n)),
            "BBB": 50 * np.cumprod(1 + rng.normal(0.0003, 0.02, n)),
            "CCC": 30 * np.cumprod(1 + rng.normal(0.0004, 0.02, n)),
        },
        index=dates,
    )


# --- 运算符 ---------------------------------------------------------------

@pytest.mark.parametrize(
    "expr",
    [
        "ts_mean(close,20)/ts_mean(close,60)-1",
        "ts_std(close/ts_delay(close,1)-1,20)",
        "-ts_pct_change(close,5)",
        "abs(ts_delta(close,10))",
        "ts_sum(close,5)/ts_sum(volume,5)",
        "clip(ts_rank(close,60),0.1,0.9)",
        "where(close>ts_mean(close,20),1,0)",
        "ts_skew(close/ts_delay(close,1)-1,20)",
        "ts_corr(close,ts_mean(close,5),20)",
    ],
)
def test_time_series_and_elementwise_operators_evaluate(expr):
    out = FactorEngine().evaluate(expr, {"close": series(), "volume": series(seed=2)})

    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == 1


@pytest.mark.parametrize(
    "expr", ["rank(close)", "zscore(ts_mean(close,20))", "quantile(close,0.5)"]
)
def test_cross_section_operators_evaluate_over_a_panel(expr):
    out = FactorEngine().evaluate(expr, {"close": panel()})

    assert out.shape == panel().shape


def test_cross_section_flag_distinguishes_the_two_modes():
    assert is_cross_sectional("rank(close)") is True
    assert is_cross_sectional("ts_mean(close,20)") is False


def test_describe_reports_variables_and_functions():
    info = FactorEngine().describe("ts_corr(close,volume,20)")

    assert set(info.variables) == {"close", "volume"}
    assert "ts_corr" in info.functions
    assert info.cross_sectional is False


# --- 安全性 ------------------------------------------------------------------

@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "close.__class__",
        "open('/etc/passwd')",
        "eval('1+1')",
        "close[0]",
        "(lambda: 1)()",
        "globals()",
        "close.__getitem__(0)",
    ],
)
def test_dangerous_expressions_are_rejected(expr):
    with pytest.raises(FactorError):
        FactorEngine().evaluate(expr, {"close": series()})


def test_window_above_the_cap_is_rejected():
    with pytest.raises(FactorError, match="窗口"):
        FactorEngine().evaluate(f"ts_mean(close,{MAX_WINDOW + 1})", {"close": series()})


def test_unknown_variable_is_rejected_with_a_message():
    with pytest.raises(FactorError, match="未知变量") as caught:
        FactorEngine().evaluate("ts_mean(price,20)", {"close": series()})
    assert "财务字段" not in str(caught.value)
    assert "pe_ttm" in str(caught.value)


def test_missing_market_variable_is_reported():
    with pytest.raises(FactorError, match="缺少变量"):
        FactorEngine().evaluate("ts_mean(close,20)", {})


def test_wrong_arity_is_rejected():
    with pytest.raises(FactorError, match="参数"):
        FactorEngine().evaluate("ts_mean(close)", {"close": series()})


def test_empty_expression_is_rejected():
    with pytest.raises(FactorError):
        FactorEngine().evaluate("   ", {"close": series()})
