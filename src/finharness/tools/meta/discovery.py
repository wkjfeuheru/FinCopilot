"""工具与技能发现：``search_tools``（docs 03.4.3）。

从模型的视角看，"这件事该用什么"是同一个问题，因此检索在同一个结果集中同时覆盖工具
与技能。底层它们仍是各自独立的注册表。

这里也是**唯一的激活入口**：检索即激活。在此之前，"发现"与"能用"之间隔着一次
``load_tool`` 调用——检索给出名字，模型还得再花一轮把它的 schema 请进来。两段式换来的
是一轮纯粹的往返开销，而注册层本来就知道全部工具。现在检索结果直接携带命中工具的
参数清单，命中的按需工具同时被激活，下一次请求里它的完整 schema 已经就位。
"""

from __future__ import annotations

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool
from finharness.tools.meta.skills import SkillRegistry


@tool(
    name="search_tools",
    description=(
        "按关键词检索可用工具与方法论技能，并激活命中的按需工具。"
        "结果中每项给出参数清单，据此可直接发起调用。"
    ),
    capability=Capability.META,
    group=ToolGroup.META,
    timeout=10,
)
class SearchToolsTool(BaseTool):
    @param("query", desc="能力描述关键词，如 '公告'、'估值'、'研报'、'ROE'")
    async def _dispatch(self, *, query: str) -> RawData:
        """检索工具与技能；激活命中的懒加载工具，并把参数清单写进结果。"""
        if self.registry is None:
            raise ValueError("当前会话未提供工具注册表")

        briefs = self.registry.search(query)
        activated = self.registry.activate_many([brief.name for brief in briefs])

        lines: list[str] = []
        for brief in briefs:
            state = "已激活" if brief.active or brief.name in activated else "未激活"
            head = f"- [工具] {brief.name}（{brief.group}，{state}）：{brief.description}"
            lines.append(head)
            if brief.inputs:
                lines.append("  参数：" + _render_inputs(brief))
            if brief.output_note:
                lines.append(f"  返回：{brief.output_note}")

        skills = SkillRegistry(self.data.settings.paths.skills_dir)
        for meta, _score in skills.search(query):
            lines.append(f"- [技能] {meta.name}：{meta.selection_hint()}")

        if not lines:
            text = "（未找到匹配的工具或技能）"
        else:
            text = "\n".join(lines)
            if activated:
                # 明说激活了什么、何时生效——否则模型无法解释为什么下一轮多出了工具，
                # 也无法判断是否不必再检索一次。
                text += (
                    "\n\n已激活：" + "、".join(activated)
                    + "（其完整说明已就位，可直接调用）"
                )

        return RawData(
            kind="text",
            text=text,
            endpoint="meta:search_tools",
            params={"query": query, "activated": activated},
        )


def _render_inputs(brief) -> str:
    """把参数清单渲染成一行：``名称（必填/可选）``。

    类型与默认值不在这里展开：模型拿到名称即可对上参数，而完整 schema 在工具被激活后
    的下一次请求里就有。这里给出的是"要传哪些名字"这一最小必要信息。
    """
    parts: list[str] = []
    for name in brief.inputs:
        parts.append(f"{name}（必填）" if name in brief.required else name)
    return "、".join(parts)
