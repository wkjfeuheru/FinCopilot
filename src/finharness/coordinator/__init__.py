"""多代理支持（docs 03.10）。

随包提供三种角色。``risk`` 是研报复核者：一次独立阅读，能重新获取一份已完成
研报背后的数据，从而捕捉到作者本人看不到的问题。``general`` 是模型可见的通用
子代理：为上下文隔离而派生，任务点名标的时可**自行取数**，任务要求时也可
``web_search``，只回传结论，因此它分析过的中间过程永远不会占用主窗口。
``reader`` 是内部角色：只消化交给它的材料、不取数；保留在协调器，
``summarize_document`` 不再内部派它（该工具只返回分片索引）。

宏观（macro）聚焦仍未实现：宏观复核产出的是叙述性判断，无法像“这个数字是否
与数据相符”那样被核验，因此这种隔离买不到任何可验证的正确性。
"""

from finharness.coordinator.reviewer import (
    GENERAL_FOCUS,
    MAX_SPAWN_TASKS,
    READER_FOCUS,
    RISK_FOCUS,
    Coordinator,
    SubAgentResult,
    focus_names,
)

__all__ = [
    "Coordinator",
    "SubAgentResult",
    "RISK_FOCUS",
    "GENERAL_FOCUS",
    "READER_FOCUS",
    "MAX_SPAWN_TASKS",
    "focus_names",
]
