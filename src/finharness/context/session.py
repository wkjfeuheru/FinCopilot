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

    # -- symbols --------------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        """Covered-symbol pool, derived from the citation registry."""
        return self.cite.resolve_symbols()

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
        """Research-state section appended to the system prompt (docs 03.6.2)."""
        parts: list[str] = []
        digest = self.plan_digest()
        if digest:
            parts.append(digest)
        if self.loaded_skills:
            parts.append("已加载方法论：" + "、".join(self.loaded_skills))
        if self.conclusions:
            recent = self.conclusions[-3:]
            rendered = "\n".join(
                f"  - {item.text}（依据 {'、'.join(item.cids) or '无'}）" for item in recent
            )
            parts.append("已形成结论：\n" + rendered)
        if not parts:
            return ""
        return "\n\n【会话研究状态】\n" + "\n".join(parts)
