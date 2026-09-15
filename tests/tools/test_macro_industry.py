"""宏观与行业数据工具：映射、渲染与 adapter 契约。"""

import asyncio
from datetime import date, timedelta

import pandas as pd

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.data.mapping import (
    MACRO_INDICATORS,
    normalize_index,
    normalize_macro_indicator,
)
from finharness.tools.fin.industry import GetIndustryConstituentsTool, GetIndustryPerfTool
from finharness.tools.fin.macro import GetMacroIndicatorsTool


class Adapter(DataAdapter):
    name = "fake"

    def fetch_macro(self, indicators, years):
        today = date.today()
        rows = []
        for slug in indicators:
            spec = MACRO_INDICATORS[slug]
            for months_back in range(3):
                moment = today - timedelta(days=30 * months_back)
                rows.append(
                    {
                        "date": pd.Timestamp(moment.replace(day=1)),
                        "indicator": slug,
                        "label": spec.label,
                        "value": 50.0 + months_back,
                        "unit": spec.unit,
                    }
                )
        return FetchResult(df=pd.DataFrame(rows), interface="fake_macro")

    def fetch_industry_perf(self, industry, years):
        if not industry:
            return FetchResult(
                df=pd.DataFrame([{"行业代码": "801010.SI", "行业名称": "农林牧渔", "成份个数": 104}]),
                interface="fake_overview",
            )
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        return FetchResult(
            df=pd.DataFrame({"index_code": "801010", "date": dates, "close": range(30)}),
            interface="fake_industry_hist",
        )

    def fetch_industry_constituents(self, industry):
        return FetchResult(
            df=pd.DataFrame({"序号": [1], "证券代码": ["000596"], "证券名称": ["古井贡酒"]}),
            interface="fake_cons",
        )


def make_access(tmp_path) -> DataAccess:
    settings = Settings(data={"cache_dir": tmp_path / "cache"}, paths={"output_dir": tmp_path / "out"})
    data = DataAccess([Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)
    data.settings = settings
    return data


# --- 映射 -----------------------------------------------------------------

def test_macro_aliases_resolve_to_slugs():
    assert normalize_macro_indicator("制造业PMI") == "pmi_manufacturing"
    assert normalize_macro_indicator("cpi_yoy") == "cpi_yoy"
    assert normalize_macro_indicator("M2同比") == "m2_yoy"


def test_unknown_macro_indicator_is_refused():
    try:
        normalize_macro_indicator("不存在")
    except ValueError as exc:
        assert "不支持的宏观指标" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown indicator should raise")


def test_index_aliases_resolve():
    assert normalize_index("沪深300") == "000300"
    assert normalize_index("000905") == "000905"


# --- 宏观工具 --------------------------------------------------------------

def test_macro_tool_renders_latest_values_and_a_series(tmp_path):
    async def run():
        return await GetMacroIndicatorsTool(make_access(tmp_path)).run(
            indicators=["pmi_manufacturing", "cpi_yoy"], years=2
        )

    result = asyncio.run(run())

    assert result.ok is True
    assert "最新值" in result.content
    assert "制造业PMI" in result.content
    assert "时序明细" in result.content


def test_macro_tool_defaults_to_a_dashboard(tmp_path):
    async def run():
        return await GetMacroIndicatorsTool(make_access(tmp_path)).run()

    result = asyncio.run(run())

    assert result.ok is True
    assert "最新值" in result.content


def test_macro_tool_rejects_an_unknown_indicator(tmp_path):
    async def run():
        return await GetMacroIndicatorsTool(make_access(tmp_path)).run(indicators=["不存在"])

    result = asyncio.run(run())

    assert result.ok is False


# --- 行业工具 ----------------------------------------------------------

def test_industry_overview_renders_when_no_industry_given(tmp_path):
    async def run():
        return await GetIndustryPerfTool(make_access(tmp_path)).run()

    result = asyncio.run(run())

    assert result.ok is True
    assert "农林牧渔" in result.content


def test_industry_perf_summarizes_an_industry_series(tmp_path):
    async def run():
        return await GetIndustryPerfTool(make_access(tmp_path)).run(industry="食品饮料", years=1)

    result = asyncio.run(run())

    assert result.ok is True
    assert "区间摘要" in result.content


def test_industry_constituents_returns_symbols(tmp_path):
    async def run():
        return await GetIndustryConstituentsTool(make_access(tmp_path)).run(industry="白酒")

    result = asyncio.run(run())

    assert result.ok is True
    assert "古井贡酒" in result.content
