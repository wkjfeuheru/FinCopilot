"""按会话维护的研究状态（docs 03.6.2）。

保存模型需要作为 *研究状态* 看到的内容：当前计划、已形成的结论、已加载的技能
与已激活的工具。原始对话记录与 token 核算仍留在 ``AgentLoop`` —— 同一事实只
放在一个地方。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from finharness.config.settings import Settings
from finharness.data.citation import CitationRegistry

STEP_STATUSES = ("pending", "done", "fail", "skipped")
_STATUS_MARK = {"pending": "○", "done": "✓", "fail": "✗", "skipped": "－"}


@dataclass(slots=True)
class PlanStep:
    seq: int
    action: str
    tool_hint: list[str] = field(default_factory=list)
    skill_hint: list[str] = field(default_factory=list)
    dep: list[int] = field(default_factory=list)
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.status not in STEP_STATUSES:
            raise ValueError(
                f"unknown step status: {self.status}; expected one of {STEP_STATUSES}"
            )


@dataclass(slots=True)
class Plan:
    plan_id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    revision: int = 1

    def progress(self) -> tuple[int, int]:
        """返回计划各步骤的 ``(已完成, 总数)``。"""
        total = len(self.steps)
        done = sum(1 for step in self.steps if step.status == "done")
        return done, total


@dataclass(slots=True)
class Conclusion:
    text: str
    cids: list[str] = field(default_factory=list)
    ts: str = ""


class ResearchContext:
    """会话级研究状态；是计划与引用视图的唯一持有者。"""

    def __init__(self, *, cite: CitationRegistry, settings: Settings) -> None:
        self.cite = cite
        self.settings = settings
        self.plan: Plan | None = None
        self.conclusions: list[Conclusion] = []
        self.loaded_skills: list[str] = []
        self.loaded_tools: list[str] = []
        # 一旦 L1 记忆建立，就由 AgentLoop 设置；这些别名转发给它，使得
        # docs 03.6.2 的 append_* 接口可用，而无需让 ctx 持有对话记录。
        self.memory: object | None = None
        # 由 loop 在对话首轮注入的记忆接口。
        # 在这里持有，是为了让每一轮渲染出相同的文本：系统 prompt 每轮会被
        # 度量三次（构建、窗口检查、压缩），这些度量结果必须一致。
        self.summary: object | None = None          # SummaryLayer
        self.short_term: object | None = None       # ShortTermMemory
        self.recalled: list[object] = []
        self.prior_conclusions: list[object] = []
        self.notes: dict[str, str] = {}
        self._remembered_symbols: list[str] = []
        # 由 loop 在对话记忆可用时设置；让元工具无需了解存储的结构即可写入
        # 全局记忆（用户偏好）。
        self.store: object | None = None
        # 会话所属用户（docs 03.13）：偏好写入按它隔离；由 loop 注入。
        self.user_id: str = ""

    # -- 对话记录转发（docs 03.6.2） -------------------------------------------
    def append_user(self, text: str) -> None:
        """转发给 L1 工作记忆；未挂载记忆时为空操作。"""
        memory = self.memory
        if memory is not None:
            memory.append_user(text)  # type: ignore[attr-defined]

    def append_tool_result(self, call_id: str, content: str) -> None:
        memory = self.memory
        if memory is not None:
            memory.append_tool_result(call_id, content)  # type: ignore[attr-defined]

    def refresh_recall(self) -> None:
        """根据对话自身的标的池重新计算召回事件。

        跳过任何已作为结论可见的内容，否则召回会浪费预算去重述屏幕上已有的事实。
        """
        short_term = self.short_term
        if short_term is None:
            return
        visible = [item.text for item in self.conclusions[-6:]]
        self.recalled = short_term.recall(  # type: ignore[attr-defined]
            self.symbols, exclude_summaries=visible
        )

    # -- 标的 ------------------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        """已覆盖标的池：引用中的标的加上任何被显式记住的标的。"""
        seen = self.cite.resolve_symbols()
        for symbol in self._remembered_symbols:
            if symbol not in seen:
                seen.append(symbol)
        return seen

    def remember_symbol(self, symbol: str) -> None:
        """记录一个已覆盖标的（用于重新加载某段对话时）。"""
        if symbol and symbol not in self._remembered_symbols:
            self._remembered_symbols.append(symbol)

    # -- 计划 ------------------------------------------------------------------
    def set_plan(self, goal: str, steps: list[PlanStep]) -> Plan:
        """安装一份计划；重复调用会修订它并递增修订号。"""
        if self.plan is not None:
            self.plan = Plan(
                plan_id=self.plan.plan_id,
                goal=goal,
                steps=steps,
                revision=self.plan.revision + 1,
            )
        else:
            self.plan = Plan(
                plan_id=f"plan_{len(self.conclusions) + 1:03d}", goal=goal, steps=steps
            )
        return self.plan

    def mark_plan_step(self, seq: int, status: str) -> bool:
        """更新某一步骤的状态；步骤不存在时返回 False。"""
        if self.plan is None:
            return False
        for step in self.plan.steps:
            if step.seq == seq:
                if status not in STEP_STATUSES:
                    raise ValueError(f"unknown step status: {status}")
                step.status = status
                return True
        return False

    def plan_digest(self) -> str:
        """渲染计划以供注入；没有计划时返回空字符串。

        每个步骤既带自身的标记，*也*带其依赖项的标记，因此模型一眼就能看出某步
        是否已解除阻塞。标题栏给出已完成/总数，形成一份紧凑的进度台账，而不是一份
        需要模型反复阅读才能弄清自己所处位置的列表。
        """
        if self.plan is None:
            return ""
        done, total = self.plan.progress()
        lines = [
            f"当前研究计划（{self.plan.plan_id}，第 {self.plan.revision} 版，"
            f"进度 {done}/{total}）：{self.plan.goal}"
        ]
        marks = {step.seq: _STATUS_MARK.get(step.status, "?") for step in self.plan.steps}
        for step in self.plan.steps:
            mark = marks.get(step.seq, "?")
            hints: list[str] = []
            if step.tool_hint:
                hints.append(f"建议工具：{', '.join(step.tool_hint)}")
            if step.skill_hint:
                hints.append(f"建议技能：{', '.join(step.skill_hint)}")
            if step.dep:
                # 渲染每个依赖项的当前标记，使未满足的前置条件无需模型
                # 交叉比对步骤编号即可看出。
                rendered = "，".join(
                    f"{dep}{marks.get(dep, '?')}" for dep in step.dep
                )
                hints.append(f"依赖：{rendered}")
            hint = f"（{'；'.join(hints)}）" if hints else ""
            lines.append(f"  {mark} {step.seq}. {step.action}{hint}")
        return "\n".join(lines)

    # -- 结论 ------------------------------------------------------------------
    def add_conclusion(self, text: str, cids: list[str]) -> Conclusion:
        """追加一条结论，附上时间戳与支撑引用的 cid。"""
        conclusion = Conclusion(
            text=text,
            cids=list(cids),
            ts=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self.conclusions.append(conclusion)
        return conclusion

    # -- 技能与工具 ------------------------------------------------------------
    def add_skill(self, name: str) -> bool:
        """记录一个已加载的技能；若此前已加载则返回 False。"""
        if name in self.loaded_skills:
            return False
        self.loaded_skills.append(name)
        return True

    def activate_tool(self, name: str) -> bool:
        """记录一个已激活的工具；若此前已激活则返回 False。"""
        if name in self.loaded_tools:
            return False
        self.loaded_tools.append(name)
        return True

    # -- 系统注入 --------------------------------------------------------------
    def state_block(self) -> str:
        """追加到系统 prompt 的研究状态段落（docs 03.6.2）。

        按模型对其需求的紧迫程度排列，组合四个不同的面向：当前计划、本对话的
        结论、召回的事件，然后是分段的历史摘要。每个面向的形态都刻意不同 —— 见
        记忆包的文档字符串。
        """
        parts: list[str] = []
        digest = self.plan_digest()
        if digest:
            parts.append(digest)
        if self.loaded_skills:
            parts.append("已加载方法论：" + "、".join(self.loaded_skills))

        conclusion_lines = self._conclusion_lines()
        if conclusion_lines:
            parts.append("本对话已形成结论：\n" + "\n".join(conclusion_lines))

        recalled = self._recalled_lines()
        if recalled:
            parts.append("相关历史事件：\n" + "\n".join(recalled))

        prior = self._prior_conclusion_lines()
        if prior:
            parts.append(
                "历史结论（来自本对话更早的轮次，时间敏感数据请重新核对）：\n"
                + "\n".join(prior)
            )

        summary = self._summary_text()
        if summary:
            parts.append(summary)

        preferences = self._preference_lines()
        if preferences:
            parts.append("用户偏好（所有对话共享）：\n" + "\n".join(preferences))

        if not parts:
            return ""
        return "\n\n【会话研究状态】\n" + "\n".join(parts)

    # -- 注入辅助 --------------------------------------------------------------
    def _conclusion_lines(self) -> list[str]:
        return [
            f"  - {item.text}（依据 {'、'.join(item.cids) or '无'}）"
            for item in self.conclusions[-3:]
        ]

    def _recalled_lines(self) -> list[str]:
        lines: list[str] = []
        for episode in self.recalled:
            kind = getattr(episode, "kind", "data")
            marker = "数据" if kind == "data" else "结论"
            lines.append(f"  - [{marker}] {getattr(episode, 'subject', '')}：{getattr(episode, 'summary', '')}")
        return lines

    def _prior_conclusion_lines(self) -> list[str]:
        """从存储中重新加载的结论；带上日期，使过期可被察觉。"""
        lines: list[str] = []
        for record in self.prior_conclusions:
            text = getattr(record, "text", "")
            if any(text == item.text for item in self.conclusions):
                continue
            formed = str(getattr(record, "ts", ""))[:10]
            subject = getattr(record, "subject", "")
            lines.append(f"  - {text}（{subject}，形成于 {formed}）")
        return lines[:5]

    def _summary_text(self) -> str:
        summary = self.summary
        if summary is None:
            return ""
        return summary.render(max_tokens=self.settings.context.ltm_inject_max_tokens)  # type: ignore[attr-defined]

    def _preference_lines(self) -> list[str]:
        return [f"  - {key}：{value}" for key, value in sorted(self.notes.items())]
