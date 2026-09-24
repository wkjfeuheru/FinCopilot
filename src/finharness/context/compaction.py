"""自动压缩：把对话中段折叠成一段摘要（docs 3.6.3）。

两条规则驱动整个设计：

* 压缩发生在轮次间隙，绝不发生在某次请求内部。
* 它绝不能阻塞对话。若摘要生成失败，回退方案改为丢弃最旧的工具结果，
  记录一条警告，然后本轮继续。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from finharness.config.settings import Settings
from finharness.context.memory.summary import SummaryLayer
from finharness.context.memory.working import KEEP_RECENT_ROUNDS, WorkingMemory
from finharness.context.tokens import CHARS_PER_TOKEN, truncate_to_tokens
from finharness.observability import NullObserver
from finharness.provider.base import Provider
from finharness.types import ModelUsage, Msg, StreamEvent

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
    seq_from: int = 0
    seq_to: int = 0
    ledger: tuple[str, ...] = ()


class AutoCompactor:
    """当下一次请求将超出窗口预算时折叠历史。"""

    def __init__(
        self,
        *,
        provider: Provider,
        memory: WorkingMemory,
        settings: Settings,
        system: str = "",
        tools: list[dict] | None = None,
        summary: SummaryLayer | None = None,
        state_text: str = "",
        observer: Any | None = None,
        on_usage: Callable[[int, int], None] | None = None,
    ) -> None:
        self.provider = provider
        self.memory = memory
        self.settings = settings
        self.system = system
        self.tools = tools or []
        # 研究状态块作为尾部消息随请求发送，而不是放在 ``system`` 中（docs 3.3），
        # 因此必须显式计入，否则窗口读数会比实际偏小。
        self.state_text = state_text
        # 若提供该字段，被折叠的历史会变成摘要分段，而不是一条合成的用户消息
        # （docs 03.6.4）。
        self.summary = summary
        # 压缩摘要本身也是一次 LLM 调用，它的 token 过去完全不计入会话成本
        # （docs 03.14.2）：这里接上观测与记账，使其以 call_type=compaction
        # 出现在指标里。
        self.observer = observer
        self.on_usage = on_usage

    def _count(self) -> int:
        return self.memory.request_tokens(
            system=self.system, tools=self.tools, extra_text=self.state_text
        )

    def needs_compaction(self) -> bool:
        """是否该压缩：token 超预算，或窗口内轮次超过硬上限。

        token 阈值只管住"窗口有多大"，管不住"有多少轮"。轮次少但每轮很小的
        对话永远触不到 token 阈值，于是历史可以无限加长——那同样是内存。轮次
        上限与 token 阈值并列，专门堵住这种长期陪伴式对话。
        """
        if self.memory.over_budget(
            system=self.system, tools=self.tools, extra_text=self.state_text
        ):
            return True
        limit = int(self.settings.context.max_window_rounds)
        return limit > 0 and self.memory.round_count() > limit

    async def compact(self) -> CompactionResult:
        """缩减窗口；失败时降级为“丢弃最旧内容”的回退方案。"""
        started = time.monotonic()
        before = self._count()
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
            summary_text = await self._summarize(foldable)
        except Exception as exc:  # noqa: BLE001 - 压缩绝不能阻塞一轮对话
            degraded = True
            warning = f"摘要生成失败，已降级为计数式摘要：{exc}"
            summary_text = self._fallback_digest(foldable)

        ledger = self._ledger(foldable)
        # 被折叠的消息所占的序号区间，从已丢弃内容之后开始。
        seq_from = self.memory.discarded + 1
        seq_to = self.memory.discarded + boundary

        removed, _discarded = self.memory.squash()
        after = self._count()

        # 当近期轮次本身很大时，仅保留它们可能仍然不够；持续收紧直到窗口
        # 回到预算之内，而不是把下一次请求留在超预算状态。
        threshold = self.settings.context.compaction_ratio * self.settings.context.context_window_tokens
        while after >= threshold:
            extra, _ = self.memory.squash(keep_rounds=1)
            if extra == 0:
                break
            removed += extra
            after = self._count()

        if self.summary is not None and removed > 0:
            self.summary.add(
                seq_from=seq_from,
                seq_to=seq_to,
                text=summary_text,
                ledger=list(ledger),
            )

        return CompactionResult(
            compacted=removed > 0,
            removed=removed,
            before_tokens=before,
            after_tokens=after,
            degraded=degraded,
            warning=warning,
            duration_ms=_ms(started),
            seq_from=seq_from if removed else 0,
            seq_to=seq_to if removed else 0,
            ledger=ledger if removed else (),
        )

    def _ledger(self, foldable: list[Msg]) -> tuple[str, ...]:
        """折叠区间内取过哪些数据的结构化记录。

        压缩会把工具结果移出窗口，因此这是模型仍能判断“某次取数已发生过”的
        唯一途径——正是它让“不要重复取数”这一约束仍然可执行。
        """
        entries: list[str] = []
        for message in foldable:
            for tool_use in message.tool_uses:
                symbol = (tool_use.args or {}).get("symbol")
                label = f"{tool_use.name}({symbol})" if symbol else tool_use.name
                if label not in entries:
                    entries.append(label)
        return tuple(entries)

    # -- 摘要生成 --------------------------------------------------------------
    async def _summarize(self, messages: list[Msg]) -> str:
        """用会话自身的 provider 执行一次摘要调用。

        provider 接口仅支持流式输出，因此把结果收集成一个字符串，
        而不是改动 provider 的契约。

        这次调用会以 ``call_type=compaction`` 记入指标，并通过 ``on_usage``
        回填给会话统计，使压缩开销在成本视图里可见而不是凭空消失。
        """
        transcript = _render_transcript(
            messages,
            counter=self.memory.counter,
            max_result_tokens=int(self.settings.context.compaction_result_tokens),
        )
        collected: list[str] = []
        usage: ModelUsage | None = None
        observer = self.observer if self.observer is not None else NullObserver()
        model = getattr(self.provider, "model", "") or ""
        async with observer.llm_span(model=model, call_type="compaction") as span:
            async for chunk in self.provider.stream(
                system=SUMMARIZE_PROMPT,
                messages=[Msg.user(transcript)],
                tools=[],
                usage=ModelUsage(),
            ):
                if chunk.event is StreamEvent.TEXT_DELTA and chunk.data:
                    collected.append(str(chunk.data))
                elif chunk.event is StreamEvent.MESSAGE_END and isinstance(
                    chunk.data, ModelUsage
                ):
                    # provider 把用量放在 MESSAGE_END 载荷里（与主循环读取的位置
                    # 一致），而不会写回传入的 usage 对象。
                    usage = chunk.data
            span.set_usage(usage)
        if self.on_usage is not None and usage is not None:
            self.on_usage(usage.input_tokens, usage.output_tokens)
        summary = "".join(collected).strip()
        if not summary:
            raise RuntimeError("摘要调用未返回任何内容")
        return summary

    def _fallback_digest(self, foldable: list[Msg]) -> str:
        """极简的替代摘要：只给计数，不做复述。

        回退方案存在的意义是缩小窗口，因此必须比它所替代的内容更小。保留每一轮
        的原文会破坏这一目的，所以只留下最初的目标与研究的大致轮廓。数据台账
        单独传递，由 ``compact`` 无论如何都会附带上去。
        """
        questions = [m.content for m in foldable if m.role == "user" and m.content]
        answers = sum(1 for m in foldable if m.role == "assistant" and m.content)
        dropped = sum(len(message.tool_results) for message in foldable)
        lines: list[str] = []
        if questions:
            goal = questions[0] or ""
            lines.append(f"起始目标：{goal[:120]}")
        lines.append(f"已完成 {answers} 轮分析与 {dropped} 次数据获取")
        if dropped:
            lines.append("数据仍可从引用中精读，无需重新获取")
        return "；".join(lines)


def _foldable_boundary(messages: list[Msg]) -> int:
    """预算所保留的轮次之前的内容都可被折叠。"""
    from finharness.context.memory.working import _recent_boundary

    return _recent_boundary(messages, KEEP_RECENT_ROUNDS)


def _render_transcript(
    messages: list[Msg],
    *,
    counter: Any | None = None,
    max_result_tokens: int = 0,
) -> str:
    """把消息列表渲染成供摘要模型阅读的纯文本对话记录。

    工具结果按 *token* 预算裁剪：这里过去是字符硬编码 ``raw[:800]``，而整个体系的其余
    部分按 token 计量，于是同一条结果在压缩转录稿里被砍得比在上下文里更短（中文下
    800 字符仅约 470 token）。给出 ``counter`` 时按真实 token 计数裁剪，与其它预算
    同一口径。
    """
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
                lines.append(f"[工具结果 {call_id}] {_clip_result(raw, counter, max_result_tokens)}")
    return "\n".join(lines)


def _clip_result(raw: str, counter: Any | None, max_tokens: int) -> str:
    """把一条工具结果裁到 token 预算内；无计数器时退回字符近似。"""
    if max_tokens <= 0:
        return raw
    if counter is None:
        return raw[: int(max_tokens * CHARS_PER_TOKEN)]
    return truncate_to_tokens(raw, counter, max_tokens, marker="…（已截断）")


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
