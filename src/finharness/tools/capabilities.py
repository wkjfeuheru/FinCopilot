"""工具能力：判定“是否用了正确的工具”所处的层级。

*能力* 指一个工具的用途（取公告、查宏观指标）。工具的各类参数与模式是同一能力内部的
*策略* —— ``detail=summary|full``、``with_text=``、``industry=None``、
``method=dcf|comps`` —— 它们都属于同一能力。在能力层级判定正确性，才能让“用错工具”
这一判断有意义，同时不惩罚合理的策略选择。

本模块是该映射的唯一权威来源。它与一个覆盖测试配套：当某个已注册工具没有对应能力时
测试会失败，因此新增工具无法悄无声息地让映射过期。
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """工具用途。当两个工具可互相替代时，它们共享同一能力；能力不同意味着
    工作本身确实不同。"""

    MARKET = "行情"
    FINANCIAL = "财务"
    VALUATION = "估值"
    PEER = "同业"
    NEWS = "新闻"
    ANNOUNCEMENT = "公告"
    RESEARCH_REPORT = "研报"
    MACRO = "宏观"
    INDUSTRY = "行业"
    COMPUTE = "计算"
    OUTPUT = "输出"
    FILE = "文件"
    WEB = "联网"
    META = "元"


# 代表研究*工作*的能力——获取或计算一批证据。只有这些才可能“对任务而言用错了”；
# 其余的是流程管道（META）、呈现（OUTPUT）或通用 I/O（FILE/WEB），它们执行任务所
# 要求的动作，因此本身永远不会出错。
RESEARCH_CAPABILITIES = frozenset(
    {
        Capability.MARKET,
        Capability.FINANCIAL,
        Capability.VALUATION,
        Capability.PEER,
        Capability.NEWS,
        Capability.ANNOUNCEMENT,
        Capability.RESEARCH_REPORT,
        Capability.MACRO,
        Capability.INDUSTRY,
        Capability.COMPUTE,
    }
)

# 每个已注册工具一条。按能力分组，使可替代的一对一眼可见
# （例如 NEWS 与 ANNOUNCEMENT——检查清单明确指出的真实混淆点）。
TOOL_CAPABILITY: dict[str, Capability] = {
    # 行情
    "get_quote": Capability.MARKET,
    "get_kline": Capability.MARKET,
    # 财务
    "get_financials": Capability.FINANCIAL,
    "get_indicators": Capability.FINANCIAL,
    # 估值
    "get_valuation": Capability.VALUATION,
    # 同业
    "get_peers": Capability.PEER,
    # 新闻 vs 公告（最典型的“用错工具”对）
    "get_market_news": Capability.NEWS,
    "get_announcements": Capability.ANNOUNCEMENT,
    # 研报
    "get_research_reports": Capability.RESEARCH_REPORT,
    # 宏观 / 行业
    "get_macro_indicators": Capability.MACRO,
    "get_industry_perf": Capability.INDUSTRY,
    "get_industry_constituents": Capability.INDUSTRY,
    # 计算
    "calc_metrics": Capability.COMPUTE,
    "calc_valuation": Capability.COMPUTE,
    "run_backtest": Capability.COMPUTE,
    # 输出
    "make_chart": Capability.OUTPUT,
    "write_report": Capability.OUTPUT,
    # 文件 / 联网
    "read_file": Capability.FILE,
    "write_file": Capability.FILE,
    "web_search": Capability.WEB,
    # 元（流程）
    "research_plan": Capability.META,
    "update_plan_step": Capability.META,
    "record_conclusion": Capability.META,
    "search_tools": Capability.META,
    "load_tool": Capability.META,
    "list_skills": Capability.META,
    "load_skill": Capability.META,
    "spawn_agent": Capability.META,
    "ask_user": Capability.META,
    "remember_preference": Capability.META,
}


class UnknownCapabilityError(KeyError):
    """查询了一个映射未覆盖的工具。"""


def capability_of(tool_name: str) -> Capability:
    """工具所属的能力。

    这里抛异常而不是返回 ``None``：未映射的工具属于映射缺陷，而静默的默认值会让
    新工具被当作永远正确。
    """
    try:
        return TOOL_CAPABILITY[tool_name]
    except KeyError as exc:  # pragma: no cover - 由覆盖测试保障
        raise UnknownCapabilityError(tool_name) from exc


def capabilities_of(tool_names: list[str]) -> set[Capability]:
    """多个工具的能力，忽略不在映射中的工具。"""
    return {
        TOOL_CAPABILITY[name] for name in tool_names if name in TOOL_CAPABILITY
    }


def is_research_capability(capability: Capability) -> bool:
    """该能力是否属于研究工作（相对于流程/呈现/IO）。"""
    return capability in RESEARCH_CAPABILITIES


# 在任务/计划文本中指示某项研究能力的关键词。用于在计划的工具提示不完整时推断其
# *意图*：一个写着“分析估值”的步骤需要估值能力，即使提示列表忘了指明工具。
# 提示只是工具名的记账；这里改为读取意图——即把同一纠正也应用于“用错工具”的判断本身。
_CAPABILITY_KEYWORDS: dict[Capability, tuple[str, ...]] = {
    Capability.MARKET: ("行情", "股价", "价格", "k线", "走势", "涨跌", "均线"),
    Capability.FINANCIAL: (
        "财务", "盈利", "营收", "收入", "利润", "roe", "毛利", "净利",
        "报表", "三表", "偿债", "现金流", "资产负债",
    ),
    Capability.VALUATION: ("估值", "市盈率", "pe", "市净率", "pb", "分位", "贵", "便宜"),
    Capability.PEER: ("同业", "可比", "对标", "可比公司"),
    Capability.NEWS: ("新闻", "资讯", "舆情"),
    Capability.ANNOUNCEMENT: ("公告", "披露", "定期报告", "临时公告"),
    Capability.RESEARCH_REPORT: ("研报", "券商报告"),
    Capability.MACRO: (
        "宏观", "经济", "通胀", "cpi", "ppi", "pmi", "gdp", "货币", "财政",
        "利率", "社融", "m2", "汇率",
    ),
    Capability.INDUSTRY: ("行业", "产业", "赛道", "产业链", "景气", "格局", "集中度"),
    Capability.COMPUTE: ("计算", "分解", "归因", "回测", "因子", "绩效"),
}


def capabilities_in_text(text: str) -> set[Capability]:
    """自由文本（目标/动作）中由关键词指示的研究能力。"""
    lowered = str(text or "").lower()
    return {
        capability
        for capability, words in _CAPABILITY_KEYWORDS.items()
        if any(word in lowered for word in words)
    }
