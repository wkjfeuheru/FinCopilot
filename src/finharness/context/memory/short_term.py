"""L2 短期记忆：用于精确召回的结构化事件。

叙述性内容属于摘要层。本层存在是为了回答叙述性内容回答不好的问题 —— “这个
我们是不是已经取过了，数据在哪里？” —— 因此它的条目刻意做得很小：一行摘要
加一个指针（cids、parquet 路径、缓存键）。把数据本身复制进来会让 L2 变成
第二个上下文窗口。

只记录两种类型，因为其他类型是冗余的：计划的状态已存在于 ``ctx.plan``，而
计算得出的值可由产生它的那次取数推导出来。
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
    """一个结构化的研究事件，小到可以保存成千上万条。"""

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
    """有界的事件环形缓冲，按标的（subject）键召回。"""

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
        """优先丢弃最旧的 *数据* 事件：它可以从缓存中恢复。

        只要还存在数据事件，就不淘汰发现与结论，因为重新取数只需一次网络调用，
        而丢弃一条结论损失的则是信息。
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
        """按标的返回最新优先的事件，跳过已可见的那些。

        ``exclude_summaries`` 携带调用方已经能看到的内容（例如从上下文中注入的
        结论），这样召回就不会浪费预算去重述屏幕上已有的事实。
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
        self,
        episodes: list[Episode],
        *,
        counter: TokenCounter,
        max_tokens: int = 0,
        header: str | None = "【相关历史事件】",
    ) -> str:
        """渲染召回的事件以供注入；未召回任何事件时返回空字符串。

        ``header`` 为 ``None`` 时只输出条目，让调用方（如会话状态块）使用自己的
        小节标题，避免同一段落里出现两个标题。
        """
        if not episodes:
            return ""
        lines = [header] if header else []
        for episode in episodes:
            marker = "数据" if episode.kind == "data" else "结论"
            lines.append(f"- [{marker}] {episode.subject}：{episode.summary}")
        text = "\n".join(lines)
        if max_tokens > 0:
            text = truncate_to_tokens(text, counter, max_tokens)
        return text
