"""情节蒸馏：把一个已完结的对话提炼成跨对话长期记忆（docs 03.6.4 LTM）。

与被作废的 L3"会话结束蒸馏"不同，这里不依赖任何优雅退出时机：蒸馏在对话
**闲置后**由后台扫描器或用户下次开新对话时补做（双保险），且绝不阻塞对话
本身——失败只累加台账计数，达到上限即放弃。

结构化写入（task_result）每轮随结论落库，不经本模块；本模块负责的是
decision / excerpt 两类需要 LLM 判断"哪些值得长期记住"的情节。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from finharness.config.settings import Settings
from finharness.context.memory.store import (
    LTM_EPISODE_KINDS,
    LTM_FACT_KINDS,
    MemoryStore,
)
from finharness.observability import NullObserver
from finharness.provider.base import Provider
from finharness.types import Msg, ModelUsage, StreamEvent

DISTILL_PROMPT = (
    "你将从一段已完成的投资研究对话中提取值得长期记住的记忆。\n"
    "输出一个 JSON 对象，包含两个数组：episodes 与 facts。\n"
    "\n"
    "episodes（情节记忆，记录\"做过什么\"）：\n"
    "- decision：用户做出的关键决策或表达的重要判断（如分析口径的取舍、关注维度的选择）；\n"
    "- excerpt：对后续研究有复用价值的对话片段（如用户的分析思路、被验证有效的做法）。\n"
    "只提取这两类；已形成的定量结论不需要（它们已按任务结果另行记录）。\n"
    "每条 summary 必须自包含：脱离原对话也能读懂，控制在 200 字以内；\n"
    "subject 填该情节涉及的标的代码（如 600519），没有则为空字符串。\n"
    "\n"
    "facts（语义记忆，记录\"知道什么\"，是去上下文的稳定知识）：\n"
    "- fact：关于某个标的/行业的稳定事实（非本对话临时算出的数字，如\"茅台属于白酒行业，受消费周期影响\"）；\n"
    "- concept：用户或团队使用的方法论、口径、概念约定（如\"这里的估值默认用 PE 而非 PB\"）；\n"
    "- preference：用户对产出形式的偏好（如\"报告要简洁，少用表格\"、\"默认看近三年\"）。\n"
    "每条的 key 是简短稳定的标识（如 industry_maotai、valuation_default、report_style），\n"
    "statement 是自包含的一句话表述；同一 key 再次出现表示更新，会覆盖旧表述。\n"
    "只提取对话中真实出现的内容；没有则给空数组。\n"
    "\n"
    "要求：不得编造；只输出 JSON 对象本身，不要任何前后缀或代码围栏。格式：\n"
    '{"episodes": [{"kind": "decision", "subject": "600519", "summary": "..."}],'
    ' "facts": [{"kind": "preference", "key": "report_style", "statement": "..."}]}'
)

# 蒸馏转录稿的体量上限：一个对话的全部消息可能非常长，而蒸馏只需要
# "发生了什么"，不需要逐字重现。按轮数截断，保留开场（研究目标）与结尾
#（最终结论），中间按预算采样——与 compaction 的转录渲染共用 token 口径。
MAX_TRANSCRIPT_TURNS = 40
MAX_RESULT_CHARS = 800


@dataclass(slots=True)
class DistillOutcome:
    """一次蒸馏的结果摘要（写入台账并可用于观测）。"""

    conversation_id: str
    episodes_written: int = 0
    facts_written: int = 0
    skipped: bool = False
    duration_ms: int = 0
    error: str | None = None


class EpisodeDistiller:
    """把单个已完结对话蒸馏为情节（decision/excerpt）与语义（fact/concept/preference）。

    两类记忆共用**同一次** LLM 调用：提示词要求一次返回 ``{episodes, facts}``。
    这不是省一次调用的微优化——它是"语义记忆默认开启也不增加成本"的前提，
    因此 ``ltm.distill_semantics=false`` 才需要显式关掉。
    """

    def __init__(
        self,
        *,
        provider: Provider,
        store: MemoryStore,
        settings: Settings,
        observer: Any | None = None,
        on_usage: Any | None = None,
        index: Any | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.settings = settings
        # 与 AutoCompactor 相同的观测接线：call_type=distill，成本可见。
        self.observer = observer
        self.on_usage = on_usage
        # 语义索引（可选）：配了 embedding 端点时为新条目写向量。
        self.index = index

    async def distill_conversation(
        self, conversation_id: str, *, user_id: str
    ) -> DistillOutcome:
        """蒸馏一个对话；任何失败都不抛出，只记入台账并返回结果。"""
        started = time.monotonic()
        record = self.store.get_conversation(conversation_id, user_id=user_id)
        if record is None:
            return DistillOutcome(
                conversation_id=conversation_id,
                skipped=True,
                duration_ms=self._ms(started),
            )
        messages = self.store.load_messages(conversation_id)
        if not messages:
            # 空对话没有可蒸馏的内容：直接记账，避免反复被扫描器选中。
            self.store.mark_ltm_distilled(conversation_id, user_id=user_id)
            return DistillOutcome(
                conversation_id=conversation_id,
                skipped=True,
                duration_ms=self._ms(started),
            )
        existing = self.store.list_ltm_episodes(
            user_id=user_id, source_conversation_id=conversation_id, limit=100
        )
        try:
            episodes, facts = await self._extract(messages, existing=existing)
            written = self._write_episodes(
                episodes, conversation_id=conversation_id, user_id=user_id,
                title=record.title or "", source_ts=record.created_at,
            )
            facts_written = self._write_facts(
                facts, conversation_id=conversation_id, user_id=user_id,
                source_ts=record.created_at,
            )
            self.store.mark_ltm_distilled(
                conversation_id, user_id=user_id, episodes_written=written
            )
            self._prune(user_id)
            return DistillOutcome(
                conversation_id=conversation_id,
                episodes_written=written,
                facts_written=facts_written,
                duration_ms=self._ms(started),
            )
        except Exception as exc:  # noqa: BLE001 - 蒸馏失败绝不致命
            state = self.store.get_ltm_distill_state(conversation_id)
            attempts = (state[0] if state else 0) + 1
            self.store.mark_ltm_distilled(
                conversation_id, user_id=user_id, attempts=attempts
            )
            return DistillOutcome(
                conversation_id=conversation_id,
                duration_ms=self._ms(started),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _write_episodes(
        self, episodes: list[dict[str, Any]], *, conversation_id: str,
        user_id: str, title: str, source_ts: str,
    ) -> int:
        written = 0
        for item in episodes:
            episode = self.store.add_ltm_episode(
                kind=item["kind"],
                summary=item["summary"],
                user_id=user_id,
                subject=str(item.get("subject") or ""),
                cids=list(item.get("cids") or []),
                source_conversation_id=conversation_id,
                source_title=title,
                source_ts=source_ts,
                distilled=True,
            )
            if episode is not None:
                written += 1
        return written

    def _write_facts(
        self, facts: list[dict[str, Any]], *, conversation_id: str,
        user_id: str, source_ts: str,
    ) -> int:
        """写入语义条目。``distill_semantics=false`` 时整体跳过（含偏好）。

        偏好也在其中：关掉语义蒸馏就意味着"不自动学习偏好"，此时偏好只能由
        ``remember_preference`` 显式写入——这与一期"不自动抽取"的口径一致，
        是留给需要严格控制的部署的开关。
        """
        if not facts or not self.settings.ltm.distill_semantics:
            return 0
        written = 0
        for item in facts:
            fact = self.store.upsert_ltm_fact(
                user_id=user_id,
                key=item["key"],
                statement=item["statement"],
                kind=item["kind"],
                subject=str(item.get("subject") or ""),
                source_conversation_id=conversation_id,
                source_ts=source_ts,
                confidence=item.get("confidence"),
            )
            if fact is None:
                continue
            written += 1
            # 向量写入是增强项：失败不影响条目本身（它仍可被键匹配召回）。
            if self.index is not None:
                try:
                    self.index.index_fact(user_id=user_id, key=fact.key)
                except Exception:  # noqa: BLE001
                    pass
        return written

    def _prune(self, user_id: str) -> None:
        """蒸馏后执行该用户的 LTM 独立保留（超限删最旧）。"""
        self.store.prune_ltm_episodes(
            user_id=user_id,
            max_episodes=int(self.settings.ltm.retention_episodes),
            max_age_days=int(self.settings.ltm.retention_days),
        )
        self.store.prune_ltm_facts(
            user_id=user_id,
            max_facts=int(self.settings.ltm.retention_facts),
            max_age_days=int(self.settings.ltm.retention_facts_days),
        )

    @staticmethod
    def _ms(started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    async def _extract(
        self, messages: list[Msg], *, existing: list[Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """调用会话 provider 提取情节与语义；产出为严格 JSON。

        provider 只有流式接口（与 AutoCompactor._summarize 同一约束），因此
        收集 TEXT_DELTA 再解析。返回 ``(episodes, facts)``。
        """
        transcript = _render_distill_transcript(messages)
        known = "\n".join(
            f"- [{item.kind}] {item.summary}" for item in existing
        ) or "（无）"
        prompt_user = (
            f"已为该对话记录的情节（避免重复提取）：\n{known}\n\n对话记录：\n{transcript}"
        )
        collected: list[str] = []
        usage: ModelUsage | None = None
        observer = self.observer if self.observer is not None else NullObserver()
        model = getattr(self.provider, "model", "") or ""
        async with observer.llm_span(model=model, call_type="distill") as span:
            async for chunk in self.provider.stream(
                system=DISTILL_PROMPT,
                messages=[Msg.user(prompt_user)],
                tools=[],
                usage=ModelUsage(),
            ):
                if chunk.event is StreamEvent.TEXT_DELTA and chunk.data:
                    collected.append(str(chunk.data))
                elif chunk.event is StreamEvent.MESSAGE_END and isinstance(
                    chunk.data, ModelUsage
                ):
                    usage = chunk.data
            span.set_usage(usage)
        if self.on_usage is not None and usage is not None:
            self.on_usage(usage.input_tokens, usage.output_tokens)
        return _parse_output("".join(collected))


def _render_distill_transcript(messages: list[Msg]) -> str:
    """把对话渲染成蒸馏模型可读的纯文本；体量有界。

    保留开场若干条（研究目标在哪）与结尾若干条（结论是什么），中间截断——
    蒸馏要的是"值得记住的判断"，不是逐字记录。
    """
    lines: list[str] = []
    for message in messages[:MAX_TRANSCRIPT_TURNS // 2]:
        lines.append(_render_line(message))
    if len(messages) > MAX_TRANSCRIPT_TURNS:
        lines.append(f"……（中间 {len(messages) - MAX_TRANSCRIPT_TURNS} 条消息已省略）……")
        for message in messages[-MAX_TRANSCRIPT_TURNS // 2 :]:
            lines.append(_render_line(message))
    return "\n".join(line for line in lines if line)


def _render_line(message: Msg) -> str:
    if message.role == "user" and message.content:
        return f"[用户] {message.content}"
    if message.role == "assistant":
        if message.content:
            return f"[助手] {message.content}"
        if message.tool_uses:
            names = "、".join(use.name for use in message.tool_uses)
            return f"[助手] （调用了工具：{names}）"
    return ""


def _strip_fence(raw: str) -> str:
    """剥掉代码围栏（部分模型仍会加上），返回其中的正文。"""
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def _json_slice(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    end = text.rfind(closer)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _parse_output(raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """解析蒸馏产出，返回 ``(episodes, facts)``。

    接受两种形态：``{"episodes": [...], "facts": [...]}``（当前提示词）与
    裸数组（旧提示词/模型未遵守格式）。后者按纯情节处理，因此升级提示词
    不会让旧形态的产出直接丢失。分支按**先出现的定界符**选择：裸数组的第
    一个 ``[`` 早于 ``{``，而对象形态反之。
    """
    text = _strip_fence(raw)
    if not text:
        return ([], [])
    obj_at = text.find("{")
    arr_at = text.find("[")
    if obj_at != -1 and (arr_at == -1 or obj_at < arr_at):
        obj = _json_slice(text, "{", "}")
        try:
            payload = json.loads(obj) if obj is not None else None
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            episodes = payload.get("episodes")
            facts = payload.get("facts")
            return (
                _valid_episodes(episodes if isinstance(episodes, list) else []),
                _valid_facts(facts if isinstance(facts, list) else []),
            )
    arr = _json_slice(text, "[", "]")
    if arr is None:
        return ([], [])
    try:
        payload = json.loads(arr)
    except json.JSONDecodeError:
        return ([], [])
    return (_valid_episodes(payload if isinstance(payload, list) else []), [])


def _parse_episodes(raw: str) -> list[dict[str, Any]]:
    """向后兼容的薄包装：只取情节部分。"""
    return _parse_output(raw)[0]


def _valid_episodes(items: list[Any]) -> list[dict[str, Any]]:
    """逐条校验情节，坏条目丢弃不致命。

    ``task_result`` 由蒸馏产出会被过滤——它由引擎每轮结构化写入，
    模型若也产出只会造成与既有条目重复的噪声。
    """
    valid: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        summary = str(item.get("summary") or "").strip()
        if kind not in LTM_EPISODE_KINDS or kind == "task_result" or not summary:
            continue
        valid.append(
            {
                "kind": kind,
                "subject": str(item.get("subject") or ""),
                "summary": summary,
                "cids": [str(cid) for cid in item.get("cids") or []],
            }
        )
    return valid


def _valid_facts(items: list[Any]) -> list[dict[str, Any]]:
    """逐条校验语义条目；``key`` 与 ``statement`` 缺一即丢弃。

    ``confidence`` 只在是数字且落在 0..1 时保留，否则交给存储层按 None 处理
    （宁可不给置信度，也不要一个伪造的精确数字）。
    """
    valid: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        key = str(item.get("key") or "").strip()
        statement = str(item.get("statement") or "").strip()
        if kind not in LTM_FACT_KINDS or not key or not statement:
            continue
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = None
        elif not 0.0 <= float(confidence) <= 1.0:
            confidence = None
        valid.append(
            {
                "kind": kind,
                "key": key,
                "statement": statement,
                "subject": str(item.get("subject") or ""),
                "confidence": confidence,
            }
        )
    return valid
