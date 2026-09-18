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


class CrowdingAdapter(DataAdapter):
    """复现真实的排序陷阱：日频序列按字母序排在最前，且行数远多于月频序列。

    适配器按 ``["indicator" 升序, "date" 降序]`` 返回，因此 ``bond_10y``（日频、上百行）
    排在 ``cpi_yoy`` / ``m2_yoy`` / ``pmi_manufacturing`` 之前。对整个长表做一次
    ``head()`` 会把月频指标完全挤出可见范围。
    """

    name = "crowding"

    def fetch_macro(self, indicators, years):
        rows = []
        for months_back in range(4):
            for slug, label, value in (
                ("cpi_yoy", "CPI同比", 0.8),
                ("m2_yoy", "M2同比", 7.5),
                ("pmi_manufacturing", "制造业PMI", 49.8),
            ):
                if slug not in indicators:
                    continue
                rows.append(
                    {
                        "date": pd.Timestamp("2026-08-01") - pd.DateOffset(months=months_back),
                        "indicator": slug,
                        "label": label,
                        "value": value - months_back,
                        "unit": MACRO_INDICATORS[slug].unit,
                    }
                )
        if "bond_10y" in indicators:
            for moment in pd.date_range(end="2026-09-17", periods=120, freq="D")[::-1]:
                rows.append(
                    {
                        "date": moment,
                        "indicator": "bond_10y",
                        "label": "10年期国债收益率",
                        "value": 2.1,
                        "unit": "%",
                    }
                )
        frame = pd.DataFrame(rows).sort_values(
            ["indicator", "date"], ascending=[True, False]
        )
        return FetchResult(df=frame.reset_index(drop=True), interface="crowding_macro")


def make_crowding_access(tmp_path) -> DataAccess:
    settings = Settings(data={"cache_dir": tmp_path / "cache"}, paths={"output_dir": tmp_path / "out"})
    data = DataAccess([CrowdingAdapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)
    data.settings = settings
    return data


def test_every_requested_indicator_survives_the_series_detail(tmp_path):
    """日频序列不得把月频指标挤出时序明细（曾经的 head() 缺陷）。

    该缺陷的真实后果是：模型看不到 PMI/M2 的历史，只能报告「未展开历史序列」，
    回答里的上月对比因此是空的——而数据其实就在手上。
    """
    async def run():
        return await GetMacroIndicatorsTool(make_crowding_access(tmp_path)).run(
            indicators=["bond_10y", "cpi_yoy", "m2_yoy", "pmi_manufacturing"]
        )

    result = asyncio.run(run())

    assert result.ok is True
    detail = result.content.split("时序明细：", 1)[1]
    for slug in ("cpi_yoy", "m2_yoy", "pmi_manufacturing", "bond_10y"):
        assert slug in detail
    # 月频序列给出多期，使上月对比可计算。
    for slug in ("cpi_yoy", "m2_yoy", "pmi_manufacturing"):
        assert detail.count(slug) >= 2


def test_series_detail_reports_per_group_omission(tmp_path):
    """被裁掉的部分必须逐组如实说明，否则「序列只有这么长」与「被裁掉」无从分辨。"""
    async def run():
        return await GetMacroIndicatorsTool(make_crowding_access(tmp_path)).run(
            indicators=["bond_10y", "cpi_yoy"]
        )

    result = asyncio.run(run())

    assert "已省略" in result.content
    assert "bond_10y" in result.content.split("时序明细：", 1)[1]


def test_macro_tool_discloses_data_period_publisher_and_cadence(tmp_path):
    """时效必须可证明：数据期、发布机构、发布节奏都要随结果给出。

    用户问「最新」而回答落在上一期时，模型需要这些事实才能区分「按月发布、当期尚未
    发布」与「系统给了旧数据」。
    """
    async def run():
        return await GetMacroIndicatorsTool(make_access(tmp_path)).run(
            indicators=["pmi_manufacturing", "cpi_yoy"]
        )

    result = asyncio.run(run())

    assert "数据时效" in result.content
    assert "中国物流与采购联合会" in result.content
    assert "国家统计局" in result.content
    assert "月度" in result.content


def test_macro_tool_marks_a_cache_hit_with_its_fetch_time(tmp_path):
    """缓存命中要能看出这份数据在本地停了多久，否则静默复用旧数据无从发现。"""
    access = make_access(tmp_path)

    async def run():
        tool = GetMacroIndicatorsTool(access)
        await tool.run(indicators=["pmi_manufacturing"])
        return await tool.run(indicators=["pmi_manufacturing"])

    second = asyncio.run(run())

    assert "缓存命中" in second.content
    assert "数据抓取时刻" in second.content


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
