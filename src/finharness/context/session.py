"""Per-session research state (docs 03.6.2).

Holds what the model needs to see as *research state*: the active plan, formed
conclusions, loaded skills and activated tools. The raw conversation transcript
and token accounting stay on ``AgentLoop`` — keeping one fact in one place.
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


@dataclass(slots=True)
class Conclusion:
    text: str
    cids: list[str] = field(default_factory=list)
    ts: str = ""


class ResearchContext:
    """Session-scoped research state; the single owner of plan and citation view."""

    def __init__(self, *, cite: CitationRegistry, settings: Settings) -> None:
        self.cite = cite
        self.settings = settings
        self.plan: Plan | None = None
        self.conclusions: list[Conclusion] = []
        self.loaded_skills: list[str] = []
        self.loaded_tools: list[str] = []
        # Set by AgentLoop once L1 memory exists; these aliases forward to it so
        # docs 03.6.2's append_* surface is available without ctx owning the
        # transcript.
        self.memory: object | None = None
        # Memory surfaces injected by the loop on the conversation's first turn.
        # Held here so every turn renders the same text: the system prompt is
        # measured three times per turn (build, window check, compaction) and
        # those measurements must agree.
        self.summary: object | None = None          # SummaryLayer
        self.short_term: object | None = None       # ShortTermMemory
        self.recalled: list[object] = []
        self.prior_conclusions: list[object] = []
        self.notes: dict[str, str] = {}
        self._remembered_symbols: list[str] = []
        # Set by the loop when conversation memory is available; lets meta tools
        # write global memory (preferences) without knowing the store's shape.
        self.store: object | None = None

    # -- transcript forwarding (docs 03.6.2) ----------------------------------
    def append_user(self, text: str) -> None:
        """Forward to L1 working memory; a no-op when memory is not attached."""
        memory = self.memory
        if memory is not None:
            memory.append_user(text)  # type: ignore[attr-defined]

    def append_tool_result(self, call_id: str, content: str) -> None:
        memory = self.memory
        if memory is not None:
            memory.append_tool_result(call_id, content)  # type: ignore[attr-defined]

    def refresh_recall(self) -> None:
        """Recompute recalled events from the conversation's own symbol pool.

        Skip anything already visible as a conclusion, or recall would spend the
        budget restating facts already on screen.
        """
        short_term = self.short_term
        if short_term is None:
            return
        visible = [item.text for item in self.conclusions[-6:]]
        self.recalled = short_term.recall(  # type: ignore[attr-defined]
            self.symbols, exclude_summaries=visible
        )

    # -- symbols --------------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        """Covered-symbol pool: citations plus any symbol explicitly remembered."""
        seen = self.cite.resolve_symbols()
        for symbol in self._remembered_symbols:
            if symbol not in seen:
                seen.append(symbol)
        return seen

    def remember_symbol(self, symbol: str) -> None:
        """Record a covered symbol (used when reloading a conversation)."""
        if symbol and symbol not in self._remembered_symbols:
            self._remembered_symbols.append(symbol)

    # -- plan -----------------------------------------------------------------
    def set_plan(self, goal: str, steps: list[PlanStep]) -> Plan:
        """Install a plan; a repeat call revises it and bumps the revision."""
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
        """Update one step's status; returns False when the step does not exist."""
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
        """Render the plan for injection; empty string when there is no plan."""
        if self.plan is None:
            return ""
        lines = [
            f"当前研究计划（{self.plan.plan_id}，第 {self.plan.revision} 版）：{self.plan.goal}"
        ]
        for step in self.plan.steps:
            mark = _STATUS_MARK.get(step.status, "?")
            hint = f"（建议工具：{', '.join(step.tool_hint)}）" if step.tool_hint else ""
            lines.append(f"  {mark} {step.seq}. {step.action}{hint}")
        return "\n".join(lines)

    # -- conclusions ----------------------------------------------------------
    def add_conclusion(self, text: str, cids: list[str]) -> Conclusion:
        conclusion = Conclusion(
            text=text,
            cids=list(cids),
            ts=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self.conclusions.append(conclusion)
        return conclusion

    # -- skills and tools -----------------------------------------------------
    def add_skill(self, name: str) -> bool:
        """Record a loaded skill; returns False when it was already loaded."""
        if name in self.loaded_skills:
            return False
        self.loaded_skills.append(name)
        return True

    def activate_tool(self, name: str) -> bool:
        """Record an activated tool; returns False when already active."""
        if name in self.loaded_tools:
            return False
        self.loaded_tools.append(name)
        return True

    # -- system injection -----------------------------------------------------
    def state_block(self) -> str:
        """Research-state section appended to the system prompt (docs 03.6.2).

        Composes four distinct surfaces, in order of how immediately the model
        needs them: the active plan, this conversation's conclusions, recalled
        events, then the segmented history summary. Each has a different shape on
        purpose — see the memory package docstrings.
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

    # -- injection helpers ----------------------------------------------------
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
        """Conclusions reloaded from the store; dated so staleness is visible."""
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
