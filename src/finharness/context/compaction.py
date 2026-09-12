"""Auto-compaction: fold the middle of a conversation into a digest (docs 3.6.3).

Two rules drive the design:

* Compaction happens in a turn gap, never inside a request.
* It must never block the conversation. If summarising fails, the fallback drops
  the oldest tool results instead, records a warning, and the turn continues.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from finharness.config.settings import Settings
from finharness.context.memory.working import KEEP_RECENT_ROUNDS, WorkingMemory
from finharness.provider.base import Provider
from finharness.types import Msg, ModelUsage, StreamEvent

SUMMARIZE_PROMPT = (
    "请把下面这段研究过程压缩成一段中文摘要，用于后续对话的上下文。要求：\n"
    "1. 保留每一步已取得的关键数据与结论（含具体数字）；\n"
    "2. 保留用过的工具与标的；\n"
    "3. 不要编造未出现的信息；\n"
    "4. 只输出摘要正文，不要任何前后缀。"
)


@dataclass(slots=True)
class CompactionResult:
    compacted: bool
    removed: int = 0
    before_tokens: int = 0
    after_tokens: int = 0
    degraded: bool = False
    warning: str | None = None
    duration_ms: int = 0


class AutoCompactor:
    """Folds history when the next request would exceed the window budget."""

    def __init__(
        self,
        *,
        provider: Provider,
        memory: WorkingMemory,
        settings: Settings,
        system: str = "",
        tools: list[dict] | None = None,
    ) -> None:
        self.provider = provider
        self.memory = memory
        self.settings = settings
        self.system = system
        self.tools = tools or []

    def needs_compaction(self) -> bool:
        return self.memory.over_budget(system=self.system, tools=self.tools)

    async def compact(self) -> CompactionResult:
        """Reduce the window, degrading to a drop-oldest fallback on failure."""
        started = time.monotonic()
        before = self.memory.request_tokens(system=self.system, tools=self.tools)
        boundary = _foldable_boundary(self.memory.raw)
        if boundary <= 0:
            return CompactionResult(
                compacted=False,
                before_tokens=before,
                after_tokens=before,
                duration_ms=_ms(started),
            )

        foldable = self.memory.raw[:boundary]
        degraded = False
        warning: str | None = None
        try:
            digest_text = await self._summarize(foldable)
        except Exception as exc:  # noqa: BLE001 - compaction must not block a turn
            degraded = True
            warning = f"摘要生成失败，已降级为丢弃最早的工具结果：{exc}"
            digest_text = self._fallback_digest(foldable)

        removed = self.memory.squash(
            digest=Msg.user(digest_text), keep_rounds=KEEP_RECENT_ROUNDS
        )
        after = self.memory.request_tokens(system=self.system, tools=self.tools)

        # Keeping the recent rounds may not be enough when those rounds are
        # themselves large; tighten until the window is back inside the budget
        # rather than leaving the next request over it.
        threshold = self.settings.context.compaction_ratio * self.settings.context.context_window_tokens
        for keep in (1, 0):
            if after < threshold:
                break
            extra = self.memory.squash(digest=Msg.user(digest_text), keep_rounds=keep)
            if extra == 0:
                break
            removed += extra
            after = self.memory.request_tokens(system=self.system, tools=self.tools)

        return CompactionResult(
            compacted=removed > 0,
            removed=removed,
            before_tokens=before,
            after_tokens=after,
            degraded=degraded,
            warning=warning,
            duration_ms=_ms(started),
        )

    # -- summarisation --------------------------------------------------------
    async def _summarize(self, messages: list[Msg]) -> str:
        """Run one summarisation call on the session's own provider.

        The provider surface is streaming-only, so the output is drained into a
        string rather than changing the provider contract.
        """
        transcript = _render_transcript(messages)
        usage = ModelUsage()
        collected: list[str] = []
        async for chunk in self.provider.stream(
            system=SUMMARIZE_PROMPT,
            messages=[Msg.user(transcript)],
            tools=[],
            usage=usage,
        ):
            if chunk.event is StreamEvent.TEXT_DELTA and chunk.data:
                collected.append(str(chunk.data))
        summary = "".join(collected).strip()
        if not summary:
            raise RuntimeError("摘要调用未返回任何内容")
        return self._digest_header(messages) + summary

    def _digest_header(self, foldable: list[Msg]) -> str:
        """Keep the traceability note: data is gone from the window, not lost."""
        citations = self.memory.ctx.cite.all() if self.memory.ctx is not None else []
        if not citations:
            return "【已压缩的历史研究过程】\n"
        cids = "、".join(item.cid for item in citations[:12])
        return (
            f"【已压缩的历史研究过程】此前已取过 {len(citations)} 份数据"
            f"（引用 {cids}），完整数据仍在缓存中，可精读或引用。\n"
        )

    def _fallback_digest(self, foldable: list[Msg]) -> str:
        """Minimal stand-in digest: counts, not a restatement.

        The fallback exists to shrink the window, so it must be smaller than what
        it replaces. Keeping every turn's text would defeat that, so only the
        original goal and the shape of the research survive.
        """
        questions = [m.content for m in foldable if m.role == "user" and m.content]
        answers = sum(1 for m in foldable if m.role == "assistant" and m.content)
        dropped = sum(len(message.tool_results) for message in foldable)
        lines = [self._digest_header(foldable).rstrip()]
        if questions:
            goal = questions[0] or ""
            lines.append(f"- 起始目标：{goal[:120]}")
        lines.append(f"- 已完成 {answers} 轮分析与 {dropped} 次数据获取")
        if dropped:
            lines.append("- 这些数据仍可从引用中精读，无需重新获取")
        return "\n".join(lines)


def _foldable_boundary(messages: list[Msg]) -> int:
    """Everything before the last KEEP_RECENT_ROUNDS rounds may be folded."""
    from finharness.context.memory.working import _recent_boundary

    return _recent_boundary(messages, KEEP_RECENT_ROUNDS)


def _render_transcript(messages: list[Msg]) -> str:
    lines: list[str] = []
    for message in messages:
        if message.role == "user" and message.content:
            lines.append(f"[用户] {message.content}")
        elif message.role == "assistant":
            if message.content:
                lines.append(f"[助手] {message.content}")
            for tool_use in message.tool_uses:
                lines.append(f"[调用工具] {tool_use.name} {tool_use.args}")
        elif message.role == "tool_result":
            for call_id, raw in message.tool_results:
                lines.append(f"[工具结果 {call_id}] {raw[:800]}")
    return "\n".join(lines)


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
