"""分段式对话摘要：L1 中“更早历史”的那一半。

压缩不会覆盖单一滚动摘要。那种做法会退化成“摘要的摘要” —— 每一轮都在重新
摘要上一轮的摘要 —— 并且最先丢失最旧的细节。相反，每次压缩都会追加一个覆盖
某段消息区间的分段，只有当这些分段合计超出预算时，才把 *最旧* 的分段合并成
一个更粗的分段。

结果是优雅地衰减，而不是突然崩塌：近期历史保持其保真度，久远历史降低分辨率，
每个分段都记录它所覆盖的区间与层级，使这种降级可被检查。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finharness.context.memory.store import MemoryStore, SummarySegment
from finharness.context.tokens import TokenCounter, truncate_to_tokens

LEDGER_CAP = 20


@dataclass
class SummaryLayer:
    """某个对话的有序摘要分段。

    ``store`` 是可选的，因此该层在没有持久化时也能工作（测试、无状态运行）：
    此时分段仅在 loop 运行期间存在于内存中，从而在不需要数据库的情况下保持
    分层行为不变。
    """

    conversation_id: str
    counter: TokenCounter
    store: MemoryStore | None = None
    budget_tokens: int = 0
    segments: list[SummarySegment] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        *,
        conversation_id: str,
        store: MemoryStore | None,
        counter: TokenCounter,
        budget_tokens: int = 0,
    ) -> SummaryLayer:
        layer = cls(
            conversation_id=conversation_id,
            counter=counter,
            store=store,
            budget_tokens=budget_tokens,
        )
        if store is not None:
            layer.segments = store.list_summary_segments(conversation_id)
        return layer

    def add(
        self, *, seq_from: int, seq_to: int, text: str, ledger: list[str] | None = None
    ) -> SummarySegment:
        """追加一个覆盖某段消息区间的分段，然后强制执行预算。"""
        if self.store is not None:
            segment = self.store.add_summary_segment(
                self.conversation_id,
                seq_from=seq_from,
                seq_to=seq_to,
                tier=0,
                text=text,
                ledger=ledger,
            )
        else:
            segment = SummarySegment(
                conversation_id=self.conversation_id,
                seq_from=int(seq_from),
                seq_to=int(seq_to),
                tier=0,
                text=text,
                ledger=tuple(ledger or []),
            )
        self.segments.append(segment)
        self.enforce_budget()
        return segment

    def tokens(self) -> int:
        """统计本层所有分段占用的 token 总数。"""
        return sum(self.counter.count(segment.text).tokens for segment in self.segments)

    def enforce_budget(self) -> int:
        """合并最旧的分段直到本层符合预算，并返回合并次数。

        合并是确定性的（先拼接，再裁剪到预算的一个份额），而不是再调用一次模型：
        粗化久远历史应当开销低廉，且绝不能失败。
        """
        merges = 0
        # 仅在剩余分段多于两个时才合并。塌缩成单个分段会连同最旧历史一起丢弃
        # 最新历史，这与本意相反；硬性上限在渲染时由 `render(max_tokens)` 施加，
        # 因此一对略微超预算的分段是可以接受的。
        while (
            self.budget_tokens > 0
            and len(self.segments) > 2
            and self.tokens() > self.budget_tokens
        ):
            first, second = self.segments[0], self.segments[1]
            # 合并后的这一对只能占用较新分段留下的空闲额度。
            remaining = sum(
                self.counter.count(segment.text).tokens for segment in self.segments[2:]
            )
            allowance = max(self.budget_tokens - remaining, 1)
            combined = SummarySegment(
                conversation_id=self.conversation_id,
                seq_from=first.seq_from,
                seq_to=second.seq_to,
                tier=max(first.tier, second.tier) + 1,
                text=truncate_to_tokens(f"{first.text}\n{second.text}", self.counter, allowance),
                ledger=_merge_ledger(first.ledger, second.ledger),
            )
            self.segments = [combined, *self.segments[2:]]
            merges += 1
        if merges and self.store is not None:
            self.store.replace_summary_segments(self.conversation_id, self.segments)
        return merges

    def ledger(self) -> tuple[str, ...]:
        """汇总各分段的取数台账，按最旧在前排序。"""
        aggregated: list[str] = []
        for segment in self.segments:
            for item in segment.ledger:
                if item not in aggregated:
                    aggregated.append(item)
        return tuple(aggregated)

    def render(self, *, max_tokens: int = 0) -> str:
        """渲染以供系统注入；尚无历史时返回空字符串。"""
        if not self.segments:
            return ""
        lines = ["【历史摘要】"]
        for segment in self.segments:
            lines.append(
                f"- （第 {segment.seq_from}-{segment.seq_to} 条消息）{segment.text}"
            )
        text = "\n".join(lines)
        if max_tokens > 0:
            text = truncate_to_tokens(text, self.counter, max_tokens)
        return text


def _merge_ledger(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    """合并两段台账并去重，只保留最近 ``LEDGER_CAP`` 条。"""
    merged = list(first)
    for item in second:
        if item not in merged:
            merged.append(item)
    return tuple(merged[-LEDGER_CAP:])
