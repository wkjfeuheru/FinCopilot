"""L2 short-term memory: structured events for precise recall.

Narrative belongs to the summary layer. This layer exists for the question
narrative answers badly — "did we already fetch this, and where is it?" — so its
entries are deliberately small: a one-line summary plus a pointer (cids, parquet
path, cache key). Copying the data in would make L2 a second context window.

Only two kinds are recorded, because the others were redundant: a plan's state
already lives in ``ctx.plan``, and a computed value is derivable from the fetch
that produced it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from finharness.context.tokens import TokenCounter, truncate_to_tokens

EPISODE_KINDS = ("data", "conclusion")
MAX_SUMMARY_CHARS = 200


@dataclass(slots=True)
class Episode:
    """One structured research event, small enough to keep thousands of."""

    kind: str
    subject: str
    summary: str
    ts: str = ""
    ref: dict[str, Any] = field(default_factory=dict)
    origin: str = "tool"

    def __post_init__(self) -> None:
        if self.kind not in EPISODE_KINDS:
            raise ValueError(f"unknown episode kind: {self.kind}; expected {EPISODE_KINDS}")
        if not self.ts:
            self.ts = datetime.now().astimezone().isoformat(timespec="seconds")
        if len(self.summary) > MAX_SUMMARY_CHARS:
            self.summary = self.summary[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"


class ShortTermMemory:
    """Bounded event ring with subject-keyed recall."""

    def __init__(self, *, cap: int = 200) -> None:
        self.cap = cap
        self._ring: list[Episode] = []

    def __len__(self) -> int:
        return len(self._ring)

    def add(self, episode: Episode) -> None:
        self._ring.append(episode)
        self._evict()

    def episodes(self) -> list[Episode]:
        return list(self._ring)

    def subjects(self) -> list[str]:
        seen: list[str] = []
        for episode in self._ring:
            if episode.subject not in seen:
                seen.append(episode.subject)
        return seen

    def _evict(self) -> None:
        """Drop the oldest *data* first: it is recoverable from the cache.

        Findings and conclusions are not evicted while a data episode remains,
        because a re-fetch costs a network call while a dropped conclusion costs
        information.
        """
        while self.cap > 0 and len(self._ring) > self.cap:
            for index, episode in enumerate(self._ring):
                if episode.kind == "data":
                    del self._ring[index]
                    break
            else:
                self._ring.pop(0)

    def recall(
        self,
        subjects: Iterable[str],
        *,
        k: int = 3,
        exclude_summaries: Iterable[str] = (),
    ) -> list[Episode]:
        """Newest-first episodes per subject, skipping ones already visible.

        ``exclude_summaries`` carries what the caller can already see (for
        example the conclusions injected from context), so recall does not spend
        budget restating a fact that is already on screen.
        """
        excluded = set(exclude_summaries)
        wanted = [subject for subject in subjects if subject]
        found: list[Episode] = []
        for subject in wanted:
            matches = [
                episode
                for episode in reversed(self._ring)
                if episode.subject == subject and episode.summary not in excluded
            ][:k]
            for episode in matches:
                if not any(
                    item.kind == episode.kind and item.summary == episode.summary
                    for item in found
                ):
                    found.append(episode)
        return found

    def render(
        self, episodes: list[Episode], *, counter: TokenCounter, max_tokens: int = 0
    ) -> str:
        """Render recalled episodes for injection; empty when nothing was recalled."""
        if not episodes:
            return ""
        lines = ["【相关历史事件】"]
        for episode in episodes:
            marker = "数据" if episode.kind == "data" else "结论"
            lines.append(f"- [{marker}] {episode.subject}：{episode.summary}")
        text = "\n".join(lines)
        if max_tokens > 0:
            text = truncate_to_tokens(text, counter, max_tokens)
        return text
