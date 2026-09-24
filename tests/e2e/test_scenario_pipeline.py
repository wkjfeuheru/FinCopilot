"""离线端到端：scenario skill -> tool -> chart，不涉及 LLM。

smoke 测试检验的是模型的路由；本文件则以确定性方式检验这些路由所依赖的
*管道*。它复现 agent 处理 quant/backtest 请求时的确切步骤序列——加载 scenario
skill、激活懒加载的 backtest tool、运行它，然后按 citation 对返回的净值帧绘制
图表——并使用 fake 数据 adapter，使其能在默认测试集中运行。

它是针对最近才被打通的两件事的回归防线：``run_backtest`` 持久化其 nav 帧，以及
``make_chart`` 叠加该帧中的多条序列。
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry, fingerprint_frame
from finharness.tools.fin.backtest import RunBacktestTool
from finharness.tools.fin.chart import MakeChartTool
from finharness.tools.fin.charting import FontUnavailableError, resolve_cjk_font
from finharness.tools.fin.macro import GetMacroIndicatorsTool
from finharness.tools.meta.skills import SkillRegistry
from finharness.tools.registry import ToolRegistry


def _prices(symbol: str, n: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    rng = np.random.default_rng(abs(hash(symbol)) % 10_000)
    close = 100 * np.cumprod(1 + rng.normal(0.0006, 0.018, n))
    return pd.DataFrame(
        {
            "date": dates, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1_000_000.0,
        }
    )


class Adapter(DataAdapter):
    name = "fake"

    def fetch_kline(self, symbol, period, adjust, years):
        return FetchResult(df=_prices(symbol), interface="fake_kline")


def _environment(tmp_path):
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"}, paths={"output_dir": tmp_path / "out"}
    )
    cite = CitationRegistry()
    data = DataAccess([Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)
    data.settings = settings
    ctx = ResearchContext(cite=cite, settings=settings)
    return settings, cite, data, ctx


def test_scenario_skill_loads_its_methodology_file(tmp_path):
    """skill 的 references 可以按名称作为文件加载，而不仅是 flow。

    加载入口现在是路由层（引擎在请求构建前调用），而不是一个工具；被测的是同一份
    ``SkillRegistry``，因此"按目标独立追踪"这一契约不变。
    """
    _settings, _cite, _data, ctx = _environment(tmp_path)
    registry = SkillRegistry(Settings().paths.skills_dir)

    async def run():
        meta, flow, _ = registry.load("quant-factor")
        _meta2, ref, _ = registry.load("quant-factor", file="references/single-series.md")
        ctx.inject_methodology("quant-factor", flow)
        ctx.inject_methodology("quant-factor/references/single-series.md", ref)
        return flow, ref

    flow, ref = asyncio.run(run())

    assert "净值" in ref or "回测" in ref
    # 两个目标在会话中被独立追踪。
    assert ctx.loaded_skills == [
        "quant-factor",
        "quant-factor/references/single-series.md",
    ]


def test_backtest_then_chart_reuses_the_persisted_nav_frame(tmp_path):
    """完整的 quant 链路：backtest -> citation -> 多序列 nav 图表。

    这才是关键的回归点：在 nav 帧被持久化之前，chart 步骤找不到它，会改为
    静默地重新获取原始价格。
    """
    try:
        resolve_cjk_font()
    except FontUnavailableError:
        import pytest

        pytest.skip("no CJK font available for chart rendering")

    settings, cite, data, ctx = _environment(tmp_path)

    result = asyncio.run(
        RunBacktestTool(data, ctx=ctx).run(
            symbol="600519", strategy="ma", params={"fast": 20, "slow": 60}, years=2, cost_bps=5
        )
    )
    assert result.ok is True, result.error
    source = result.sources[0]
    # 该帧可被绘制成图，因为它携带了 parquet 路径。
    assert source.parquet_path, "backtest frame must be persisted for charting"
    assert {"date", "nav", "benchmark_nav"} <= set(source.df.columns)

    # 按 loop 的方式注册 citation，然后按 cid 绘制图表。
    cid = cite.register(
        tool="run_backtest",
        endpoint=source.endpoint,
        symbol="600519",
        params=source.params,
        rows=len(source.df),
        cols=len(source.df.columns),
        fingerprint=fingerprint_frame(source.df),
        parquet_path=source.parquet_path,
    ).cid

    chart = asyncio.run(
        MakeChartTool(data, ctx=ctx).run(
            type="line", title="净值对比", cids=[cid], series=["nav", "benchmark_nav"]
        )
    )
    assert chart.ok is True, chart.error
    assert chart.sources[0].params["series"] == ["nav", "benchmark_nav"]
    png = chart.attachments[0]
    from pathlib import Path

    assert Path(png).is_file()


def test_macro_tool_is_reachable_and_grounded(tmp_path):
    """macro scenario 的数据路由返回真实形态的长格式序列。"""

    class MacroAdapter(DataAdapter):
        name = "fake_macro"

        def fetch_macro(self, indicators, years):
            rows = [
                {
                    "date": pd.Timestamp("2026-08-01"),
                    "indicator": slug,
                    "label": slug,
                    "value": 50.0,
                    "unit": "指数",
                }
                for slug in indicators
            ]
            return FetchResult(df=pd.DataFrame(rows), interface="fake_macro")

    settings = Settings(
        data={"cache_dir": tmp_path / "cache"}, paths={"output_dir": tmp_path / "out"}
    )
    data = DataAccess([MacroAdapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)
    data.settings = settings
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)

    result = asyncio.run(
        GetMacroIndicatorsTool(data, ctx=ctx).run(indicators=["pmi_manufacturing"], years=2)
    )

    assert result.ok is True, result.error
    assert "最新值" in result.content


def test_registry_exposes_the_new_tools_with_correct_tiers(tmp_path):
    """常驻/懒加载划分：backtest 为懒加载，macro/industry 数据为常驻。"""
    settings = Settings()
    registry = ToolRegistry(DataAccess([], settings=settings), settings=settings)

    assert "run_backtest" in registry.lazy_names()
    assert "get_macro_indicators" in registry.resident_names()
    assert "get_industry_perf" in registry.resident_names()
    assert "get_industry_constituents" in registry.resident_names()
    # 懒加载工具同样可解析：激活只是何时注入 schema，不是能否触达。
    assert registry.resolve("run_backtest") is not None
