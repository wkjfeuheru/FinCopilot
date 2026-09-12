"""Memory layers (docs 03.6.4).

* L1 working memory — the raw transcript and its accounting.
* Summary layer — segmented, merged summaries of earlier history.
* L2 short-term memory — structured events for precise recall.
* Persistent store — conversation transcript, citations, conclusions, notes.
"""

from finharness.context.memory.short_term import Episode, ShortTermMemory
from finharness.context.memory.store import (
    ConclusionRecord,
    ConversationRecord,
    MemoryStore,
    SummarySegment,
)
from finharness.context.memory.summary import SummaryLayer
from finharness.context.memory.working import WORKING_WINDOW_FLOOR, WindowUsage, WorkingMemory

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
