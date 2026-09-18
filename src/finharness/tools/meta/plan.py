"""research_plan：安装或修订会话计划（docs 03.6.2）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from finharness.context.session import PlanStep
from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool


class PlanStepInput(BaseModel):
    """计划步骤。

    ``sections`` 式的嵌套结构交给一个模型（``@param(annotation=...)``），因为它是
    ``research_plan`` 的输入**形态**而非工具的散装参数；``tool_hint``/``skill_hint``
    仍是模型对执行的建议，路由层会读它们。
    """

    seq: int = Field(description="步骤序号，从 1 开始")
    action: str = Field(description="这一步要做什么")
    tool_hint: list[str] = Field(default_factory=list, description="建议使用的工具名")
    skill_hint: list[str] = Field(default_factory=list, description="建议加载的技能名")
    dep: list[int] = Field(default_factory=list, description="依赖的前置步骤序号")


@tool(
    name="research_plan",
    description=(
        "为复杂多步研究问题制定执行计划；简单事实问题不需要调用。"
        "再次调用会修订计划并递增版本号。"
    ),
    capability=Capability.META,
    group=ToolGroup.META,
    timeout=30,
)
class ResearchPlanTool(BaseTool):
    @param("goal", desc="研究目标，一句话")
    @param("steps", annotation=list[PlanStepInput], desc="有序步骤列表")
    async def _dispatch(self, *, goal: str, steps: list[dict]) -> RawData:
        """解析步骤并在研究上下文中落计划，返回计划摘要。"""
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


_STATUS = Literal["pending", "done", "fail", "skipped"]


@tool(
    name="update_plan_step",
    description=(
        "回写研究计划中某一步的执行状态（完成/失败/跳过）；"
        "每完成或放弃一个步骤后调用，使计划进度如实反映进展。"
    ),
    capability=Capability.META,
    group=ToolGroup.META,
    timeout=30,
)
class UpdatePlanStepTool(BaseTool):
    """记录某一步的结果，使计划反映实际发生的情况。

    没有它，计划就是一份永不推进的文档：摘要会一直把每一步渲染为 pending。把状态
    回写，才使它成为模型（与用户）可以信赖的进度台账。
    """

    @param("seq", desc="要更新的步骤序号（research_plan 中声明的 seq）")
    @param(
        "status",
        annotation=_STATUS,
        desc="该步骤的新状态：done 完成、fail 失败、skipped 跳过、pending 重置",
    )
    async def _dispatch(self, *, seq: int, status: str) -> RawData:
        """回写指定步骤的状态，并返回更新后的计划摘要。"""
        if self.ctx is None:
            raise ValueError("当前会话未启用研究上下文，无法更新计划")
        if not self.ctx.mark_plan_step(seq, status):
            raise ValueError(f"计划中没有步骤 {seq}；请核对序号或先用 research_plan 修订计划")
        return RawData(
            kind="text",
            text=self.ctx.plan_digest(),
            endpoint="meta:update_plan_step",
            params={"seq": seq, "status": status},
        )


@tool(
    name="record_conclusion",
    description=(
        "记录一条已形成的结论及其依据（citations）；"
        "会在研究状态中回显、并随会话持久化，供后续轮次与重开对话复用。"
    ),
    capability=Capability.META,
    group=ToolGroup.META,
    timeout=30,
)
class RecordConclusionTool(BaseTool):
    """持久化已形成的结论，使其在压缩与重载后仍然保留。

    在此记录的结论会被渲染进会话状态块并写入会话存储，因此即便先前的工具结果被压缩
    掉，长任务仍能保留已确立的事实。
    """

    @param("text", desc="一句话结论，须是可复述的事实性判断")
    @param("cids", desc="支撑该结论的 citation id 列表，如 cit_000001")
    async def _dispatch(self, *, text: str, cids: list[str]) -> RawData:
        """将结论及其引用 id 写入研究上下文，并返回回显文本。"""
        if self.ctx is None:
            raise ValueError("当前会话未启用研究上下文，无法记录结论")
        conclusion = self.ctx.add_conclusion(text.strip(), list(cids))
        return RawData(
            kind="text",
            text=f"已记录结论：{conclusion.text}（依据 {'、'.join(conclusion.cids) or '无'}）",
            endpoint="meta:record_conclusion",
            params={"cids": len(cids)},
        )

