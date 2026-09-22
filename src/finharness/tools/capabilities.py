"""工具能力：判定“是否用了正确的工具”所处的层级。

*能力* 指一个工具的用途（取公告、查宏观指标）。工具的各类参数与模式是同一能力内部的
*策略* —— ``detail=summary|full``、``with_text=``、``industry=None``、
``method=dcf|comps`` —— 它们都属于同一能力。在能力层级判定正确性，才能让“用错工具”
这一判断有意义，同时不惩罚合理的策略选择。

能力不再由本模块手工维护：它是 ``@tool`` 声明的一部分（``declare.py`` 的 ``ToolSpec``）。
此前这里有一份 ``TOOL_CAPABILITY`` 映射必须与工具清单手工同步，漏登记的后果是新工具
静默地"永远用对"；现在漏声明 ``capability`` 会让工具无法通过 ``@tool``，在导入期即失败。
本模块保留的是**按其能力使用它**的推理部分：哪些能力代表真正的研究工作、以及如何从
任务文本推断意图。
"""

from __future__ import annotations

import re

from finharness.tools.declare import (
    DECLARED_TOOLS,
    Capability,
    DeclarationError,
    ToolSpec,
)

__all__ = [
    "Capability",
    "KEYWORDS_BY_CAPABILITY",
    "RESEARCH_CAPABILITIES",
    "UnknownCapabilityError",
    "capabilities_in_text",
    "capabilities_of",
    "capability_of",
    "is_research_capability",
    "spec_of",
]


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
        # 数据集派发器取的是证据（特色数据、基金、期货、期权），因此与上面几项同类：
        # 它是"研究工作"，也就可能对某个任务而言用错了。FIN_DATA 分组的工具必须落在
        # 这个集合里，否则"计划指向 A 数据、实际取了 B 数据"就失去唯一的判定层。
        Capability.DATASET,
        Capability.COMPUTE,
    }
)


class UnknownCapabilityError(KeyError):
    """查询了一个没有声明的工具。"""


def spec_of(tool_name: str) -> ToolSpec:
    """按名称取回工具声明。

    工具名是声明的一部分，因此查不到即说明该名称从未被声明过——抛出异常而不是返回
    ``None``，理由与 ``declare.declared`` 相同：静默的默认值会让一个不存在的能力
    继续被当作正常值使用。
    """
    try:
        return DECLARED_TOOLS[tool_name]
    except KeyError as exc:
        raise UnknownCapabilityError(tool_name) from exc


def capability_of(tool_name: str) -> Capability:
    """工具所属的能力（来自其声明）。"""
    try:
        return spec_of(tool_name).capability
    except DeclarationError as exc:  # pragma: no cover - 由声明覆盖测试保障
        raise UnknownCapabilityError(tool_name) from exc


def capabilities_of(tool_names: list[str]) -> set[Capability]:
    """多个工具的能力，忽略没有声明的名称。

    计划里的 ``tool_hint`` 可能带有过时或拼错的工具名，那是模型输入而非映射缺陷，
    因此这里容忍未知项而不是让整次进度判定失败。
    """
    return {
        DECLARED_TOOLS[name].capability
        for name in tool_names
        if name in DECLARED_TOOLS
    }


def is_research_capability(capability: Capability) -> bool:
    """该能力是否属于研究工作（相对于流程/呈现/IO）。"""
    return capability in RESEARCH_CAPABILITIES


# 在任务/计划文本中指示某项研究能力的关键词。用于在计划的工具提示不完整时推断其
# *意图*：一个写着“分析估值”的步骤需要估值能力，即使提示列表忘了指明工具。
# 提示只是工具名的记账；这里改为读取意图——即把同一纠正也应用于“用错工具”的判断本身。
#
# 路由层复用同一张表，把用户消息推断为需要哪些能力，据此决定注入哪些方法论
# （见 ``tools/meta/skills.py`` 的 ``route``）。两处共用一张表，是为了让"计划认为该用
# 什么"与"引擎认为该注入什么"永远建立在同一个意图判定上。
_CAPABILITY_KEYWORDS: dict[Capability, tuple[str, ...]] = {
    Capability.MARKET: ("行情", "股价", "价格", "k线", "走势", "涨跌", "均线"),
    Capability.FINANCIAL: (
        "财务", "盈利", "营收", "收入", "利润", "roe", "毛利", "净利",
        "报表", "三表", "偿债", "现金流", "资产负债",
    ),
    # "贵" 不能作为关键词单独出现：它出现在"贵州茅台"这类**公司名**里，于是每一句问
    # 茅台的话都会被读成估值问题（曾经如此）。单字判断一律拆成不与其混淆的多字形式。
    Capability.VALUATION: (
        "估值", "市盈率", "pe", "市净率", "pb", "分位",
        "高估", "低估", "偏贵", "太贵", "贵不贵", "便宜",
    ),
    Capability.PEER: ("同业", "可比", "对标", "可比公司"),
    Capability.NEWS: ("新闻", "资讯", "舆情"),
    Capability.ANNOUNCEMENT: ("公告", "披露", "定期报告", "临时公告"),
    Capability.RESEARCH_REPORT: ("研报", "券商报告"),
    Capability.MACRO: (
        "宏观", "经济", "通胀", "cpi", "ppi", "pmi", "gdp", "货币", "财政",
        "利率", "社融", "m2", "汇率",
    ),
    Capability.INDUSTRY: ("行业", "产业", "赛道", "产业链", "景气", "格局", "集中度"),
    # 长尾数据集：特色数据、基金、期货、期权、交易日历。这些是"另有一类东西
    # 可查"的信号，而不是某类研究工作——当问题落在这些域里时，路由层应注入数据集
    # 派发器，而不是去激活一个语义上更近但答不了它的工具。
    Capability.DATASET: (
        "涨停", "跌停", "炸板", "连板", "龙虎榜", "游资", "热股", "人气榜",
        "异动", "集合竞价", "竞价", "交易日", "交易日历",
        "基金", "净值", "基金经理", "重仓", "持仓",
        "期货", "合约", "仓单", "基差", "期权", "ETF", "LOF", "REITs",
        "概念板块", "概念指数",
    ),
    Capability.COMPUTE: ("计算", "分解", "归因", "回测", "因子", "绩效"),
}

# 按能力反查其关键词，供路由层按能力取用（而不是再抄一份关键词表）。
KEYWORDS_BY_CAPABILITY = _CAPABILITY_KEYWORDS

# 拉丁关键词两侧不得再是字母或数字。中文没有空白分词，直接子串匹配（"ROE差异"、
# "PE(TTM)" 都要命中），但纯拉丁缩写若也按子串匹配就会在别的词内部误命中——
# "pe" 落在 "type" 里、"m2" 落在 "form2" 里。以"两侧不是字母数字"刻画边界，
# 正好兼顾两种情形：紧跟中文或标点算命中，嵌在英文单词里不算。
_LATIN_EDGE = re.compile(r"[a-z0-9]")


def _matches(keyword: str, lowered: str) -> bool:
    """关键词是否命中。

    拉丁关键词要求边界（见 ``_LATIN_EDGE``），中文关键词直接子串匹配。
    """
    if not keyword.isascii():
        return keyword in lowered
    start = lowered.find(keyword)
    while start != -1:
        end = start + len(keyword)
        left_ok = start == 0 or not _LATIN_EDGE.match(lowered[start - 1])
        right_ok = end >= len(lowered) or not _LATIN_EDGE.match(lowered[end])
        if left_ok and right_ok:
            return True
        start = lowered.find(keyword, start + 1)
    return False


def capabilities_in_text(text: str) -> set[Capability]:
    """自由文本（目标/动作）中由关键词指示的研究能力。"""
    lowered = str(text or "").lower()
    return {
        capability
        for capability, words in _CAPABILITY_KEYWORDS.items()
        if any(_matches(word, lowered) for word in words)
    }
