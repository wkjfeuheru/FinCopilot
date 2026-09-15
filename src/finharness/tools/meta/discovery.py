"""工具与技能发现：search_tools、load_tool（docs 03.4.3）。

从模型的视角看，“这件事该用什么”是同一个问题，因此检索在同一个结果集中同时覆盖工具
与技能。底层它们仍是各自独立的注册表。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup
from finharness.tools.meta.skills import SkillRegistry


class SearchToolsInput(BaseModel):
    query: str = Field(description="能力描述关键词，如 '公告'、'估值'、'研报'、'ROE'")


class SearchToolsTool(BaseTool):
    name = "search_tools"
    description = "按关键词检索可用工具与方法论技能（含尚未激活的懒加载工具）。"
    input_model = SearchToolsInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self, *, query: str) -> RawData:
        """按关键词同时检索工具与技能，合并为带激活状态的文本列表。"""
        if self.registry is None:
            raise ValueError("当前会话未提供工具注册表")

        lines: list[str] = []
        for brief in self.registry.search(query):
            state = "已激活" if self.registry.is_active(brief.name) else "未激活"
            lines.append(f"- [工具] {brief.name}（{brief.group}，{state}）：{brief.description}")

        skills = SkillRegistry(self.data.settings.paths.skills_dir)
        for meta, _score in skills.search(query):
            lines.append(f"- [技能] {meta.name}：{meta.selection_hint()}")

        text = "\n".join(lines) if lines else "（未找到匹配的工具或技能）"
        return RawData(
            kind="text", text=text, endpoint="meta:search_tools", params={"query": query}
        )


class LoadToolInput(BaseModel):
    name: str = Field(description="要激活的工具名（来自 search_tools 结果中的 [工具] 项）")


class LoadToolTool(BaseTool):
    name = "load_tool"
    description = "激活一个懒加载工具，使其在下一轮可以被调用。"
    input_model = LoadToolInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self, *, name: str) -> RawData:
        """在注册表与上下文中激活指定懒加载工具，使其下一轮可被调用。"""
        if self.registry is None:
            raise ValueError("当前会话未提供工具注册表")
        if self.registry.resolve(name) is None:
            raise ValueError(f"未找到工具：{name}")
        already = self.registry.is_active(name)
        if not already:
            self.registry.activate(name)
        if self.ctx is not None:
            self.ctx.activate_tool(name)
        # schema 会在*下一次*请求中出现，符合“模型只能调用已获得其 schema 的工具”。
        text = (
            f"工具 {name} 已在可用列表中。"
            if already
            else f"已激活工具 {name}，将在下一轮可调用。"
        )
        return RawData(kind="text", text=text, endpoint="meta:load_tool", params={"name": name})
