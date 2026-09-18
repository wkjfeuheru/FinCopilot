"""capability 映射：每个已注册 tool 都被覆盖，且易混淆的
配对确实是不同的 capability。"""

import pytest

from finharness.tools.base import ToolGroup
from finharness.tools.capabilities import (
    RESEARCH_CAPABILITIES,
    Capability,
    UnknownCapabilityError,
    capabilities_of,
    capability_of,
    is_research_capability,
)
from finharness.tools.declare import DECLARED_TOOLS
from finharness.tools.registry import ALL_TOOL_CLASSES


def test_every_registered_tool_has_a_capability():
    """每个已注册工具都必须带一条声明。

    覆盖门禁的形态变了、目的没变：过去是"新工具必须在 ``TOOL_CAPABILITY`` 里多写一行"，
    现在是"新工具必须经 ``@tool`` 声明 capability"。漏声明的工具会在导入期失败，而这条
    断言守着另一头——注册表里的类必须真的被声明过。
    """
    registered = {cls.name for cls in ALL_TOOL_CLASSES}
    missing = registered - set(DECLARED_TOOLS)
    assert not missing, f"工具缺少声明：{sorted(missing)}"


def test_every_class_declares_under_its_own_name():
    """类名与声明名必须一致。

    这条取代了旧的"映射表没有过期条目"：那份手工映射已经不存在了（声明与类同处一地，
    不可能各自漂移），剩下来真正会出错的是**两者写得不一致**——``@tool(name=...)`` 与类
    的注册名不同，会让 ``DECLARED_TOOLS`` 与 ``ALL_TOOL_CLASSES`` 各说一套。
    """
    for cls in ALL_TOOL_CLASSES:
        spec = getattr(cls, "__tool_spec__", None)
        assert spec is not None, f"{cls.__name__} 未经 @tool 声明"
        assert spec.name == cls.name, f"{cls.__name__}: 声明名 {spec.name} != 类名 {cls.name}"
        assert DECLARED_TOOLS[cls.name] is spec, cls.name


def test_no_two_classes_share_a_declared_name():
    names = [cls.name for cls in ALL_TOOL_CLASSES]
    assert len(names) == len(set(names)), "工具名重复会让注册表静默覆盖其中一个"


def test_every_declared_capability_is_an_enum_member():
    """声明里的 capability 必须是枚举成员，而不是恰好同名的字符串。"""
    for name, spec in DECLARED_TOOLS.items():
        assert isinstance(spec.capability, Capability), name


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
                 "search_tools", "research_plan"):
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
