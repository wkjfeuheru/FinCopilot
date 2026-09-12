"""Segmented conversation summaries: the "earlier history" half of L1.

Compaction does not overwrite one rolling digest. That degrades into a summary of
a summary — each pass re-summarising the previous one — and loses the oldest
detail first. Instead every compaction appends a segment covering a message
range, and only when the segments collectively exceed their budget are the
*oldest* ones merged into a coarser segment.

The result is graceful decay rather than sudden collapse: recent history keeps
its fidelity, distant history loses resolution, and each segment records the
range and tier it covers so the degradation is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finharness.context.memory.store import MemoryStore, SummarySegment
from finharness.context.tokens import TokenCounter, truncate_to_tokens

LEDGER_CAP = 20


@dataclass
class SummaryLayer:
    """Ordered summary segments for one conversation.

    ``store`` is optional so the layer also works without persistence (tests,
    stateless runs): segments then live only in memory for the duration of the
    loop, which keeps the layered behaviour intact without requiring a database.
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
    ) -> "SummaryLayer":
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
        """Append a segment covering a message range, then enforce the budget."""
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
        return sum(self.counter.count(segment.text).tokens for segment in self.segments)

    def enforce_budget(self) -> int:
        """Merge the oldest segments until the layer fits, returning merges made.

        Merging is deterministic (concatenate, then trim to a share of the
        budget) rather than another model call: coarsening distant history should
        be cheap and must not be able to fail.
        """
        merges = 0
        # Merge only while more than two segments remain. Collapsing to a single
        # segment would discard the newest history along with the oldest, which
        # is the opposite of the intent; the hard cap is applied at render time
        # by `render(max_tokens)`, so a slightly over-budget pair is fine.
        while (
            self.budget_tokens > 0
            and len(self.segments) > 2
            and self.tokens() > self.budget_tokens
        ):
            first, second = self.segments[0], self.segments[1]
            # The merged pair may only take what the newer segments leave free.
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
        """Aggregate the data-fetch ledger across segments, oldest first."""
        aggregated: list[str] = []
        for segment in self.segments:
            for item in segment.ledger:
                if item not in aggregated:
                    aggregated.append(item)
        return tuple(aggregated)

    def render(self, *, max_tokens: int = 0) -> str:
        """Render for system injection; empty when there is no history yet."""
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
    merged = list(first)
    for item in second:
        if item not in merged:
            merged.append(item)
    return tuple(merged[-LEDGER_CAP:])
