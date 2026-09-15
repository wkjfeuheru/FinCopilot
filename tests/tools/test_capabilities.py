"""capability 映射：每个已注册 tool 都被覆盖，且易混淆的
配对确实是不同的 capability。"""

import pytest

from finharness.tools.base import ToolGroup
from finharness.tools.capabilities import (
    RESEARCH_CAPABILITIES,
    TOOL_CAPABILITY,
    Capability,
    UnknownCapabilityError,
    capabilities_of,
    capability_of,
    is_research_capability,
)
from finharness.tools.registry import ALL_TOOL_CLASSES


def test_every_registered_tool_has_a_capability():
    """新增 tool 却无 capability = 映射已过期，这正是先前硬编码
    白名单腐坏的方式。改为大声失败。"""
    registered = {cls.name for cls in ALL_TOOL_CLASSES}
    missing = registered - set(TOOL_CAPABILITY)
    assert not missing, f"工具缺少能力映射：{sorted(missing)}"


def test_capability_map_has_no_stale_entries():
    registered = {cls.name for cls in ALL_TOOL_CLASSES}
    extra = set(TOOL_CAPABILITY) - registered
    assert not extra, f"能力映射引用了不存在的工具：{sorted(extra)}"


def test_announcements_and_news_are_different_capabilities():
    """checklist 标记的典型可替代项：它们绝不能坍缩为同一项。"""
    assert capability_of("get_announcements") is Capability.ANNOUNCEMENT
    assert capability_of("get_market_news") is Capability.NEWS
    assert capability_of("get_announcements") is not capability_of("get_market_news")


def test_tools_that_substitute_share_a_capability():
    # quote 与 kline 都回答"价格"；需要价格的任务由任一个来满足。
    assert capability_of("get_quote") is capability_of("get_kline")
    # financials 与 indicators 都属于公司财务。
    assert capability_of("get_financials") is capability_of("get_indicators")
    # 行业 tools 属于同一个 capability。
    assert capability_of("get_industry_perf") is capability_of("get_industry_constituents")


def test_macro_and_industry_are_distinct():
    assert capability_of("get_macro_indicators") is Capability.MACRO
    assert capability_of("get_industry_perf") is Capability.INDUSTRY


def test_research_capabilities_exclude_process_and_presentation():
    for name in ("make_chart", "write_report", "read_file", "web_search",
                 "load_skill", "research_plan"):
        assert not is_research_capability(capability_of(name)), name


def test_research_capabilities_include_the_data_and_compute_tools():
    for name in ("get_announcements", "get_market_news", "get_macro_indicators",
                 "run_backtest", "calc_metrics"):
        assert is_research_capability(capability_of(name)), name


def test_unknown_tool_raises_rather_than_defaulting():
    with pytest.raises(UnknownCapabilityError):
        capability_of("does_not_exist")


def test_capabilities_of_ignores_unknown_names():
    """Plan hints 可能引用目录中不存在的 tool（过期 hint）；
    该 helper 不得因此崩溃。"""
    result = capabilities_of(["get_announcements", "not_a_tool"])

    assert result == {Capability.ANNOUNCEMENT}


def test_group_and_capability_are_consistent_for_fin_data():
    """每个 FIN_DATA tool 都映射到某个 research capability（健全性检查，而非策略）。"""
    for cls in ALL_TOOL_CLASSES:
        if cls.group is ToolGroup.FIN_DATA:
            assert is_research_capability(capability_of(cls.name)), cls.name
