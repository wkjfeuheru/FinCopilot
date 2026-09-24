"""记忆层（docs 03.6.4）。

* L1 工作记忆 —— 原始对话记录及其核算。
* 摘要层 —— 对更早历史的分段、合并后的摘要。
* L2 短期记忆 —— 用于精确召回的结构化事件。
* 持久化存储 —— 对话记录、引用、结论、笔记。
"""

from finharness.context.memory.records import (
    ConclusionRecord,
    ConversationRecord,
    SummarySegment,
)
from finharness.context.memory.short_term import Episode, ShortTermMemory
from finharness.context.memory.store import MemoryStore
from finharness.context.memory.summary import SummaryLayer
from finharness.context.memory.working import (
    WORKING_WINDOW_FLOOR,
    WindowUsage,
    WorkingMemory,
)

__all__ = [
    "ConclusionRecord",
    "ConversationRecord",
    "Episode",
    "MemoryStore",
    "ShortTermMemory",
    "SummaryLayer",
    "SummarySegment",
    "WORKING_WINDOW_FLOOR",
    "WindowUsage",
    "WorkingMemory",
]
