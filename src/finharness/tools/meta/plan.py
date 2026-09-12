"""research_plan: install or revise the session plan (docs 03.6.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.context.session import PlanStep
from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class PlanStepInput(BaseModel):
    seq: int = Field(description="步骤序号，从 1 开始")
    action: str = Field(description="这一步要做什么")
    tool_hint: list[str] = Field(default_factory=list, description="建议使用的工具名")
    skill_hint: list[str] = Field(default_factory=list, description="建议加载的技能名")
    dep: list[int] = Field(default_factory=list, description="依赖的前置步骤序号")


class ResearchPlanInput(BaseModel):
    goal: str = Field(description="研究目标，一句话")
    steps: list[PlanStepInput] = Field(description="有序步骤列表")


class ResearchPlanTool(BaseTool):
    name = "research_plan"
    description = (
        "为复杂多步研究问题制定执行计划；简单事实问题不需要调用。"
        "再次调用会修订计划并递增版本号。"
    )
    input_model = ResearchPlanInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 30

    async def _dispatch(self, *, goal: str, steps: list[dict]) -> RawData:
        if self.ctx is None:
            raise ValueError("当前会话未启用研究上下文，无法落计划")
        parsed = [PlanStep(**step) for step in steps]
        plan = self.ctx.set_plan(goal, parsed)
        return RawData(
            kind="text",
            text=self.ctx.plan_digest(),
            endpoint="meta:research_plan",
            params={"plan_id": plan.plan_id, "revision": plan.revision, "steps": len(parsed)},
        )
