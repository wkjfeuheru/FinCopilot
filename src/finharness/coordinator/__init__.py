"""多代理支持（docs 03.10）。

随包提供两种角色。``risk`` 是研报复核者：一次独立阅读，能重新获取一份已完成
研报背后的数据，从而捕捉到作者本人看不到的问题。``general`` 是纯粹为上下文
隔离而派生的 worker——一个任务在它自己的上下文中运行至完成，只回传结论，
因此它消化过的材料永远不会占用主窗口。

宏观（macro）聚焦仍未实现：宏观复核产出的是叙述性判断，无法像“这个数字是否
与数据相符”那样被核验，因此这种隔离买不到任何可验证的正确性。
"""

from finharness.coordinator.reviewer import (
    GENERAL_FOCUS,
    MAX_SPAWN_TASKS,
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
    "MAX_SPAWN_TASKS",
    "focus_names",
]
