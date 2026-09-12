"""Tool discovery and activation: search_tools, load_tool (docs 03.4.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class SearchToolsInput(BaseModel):
    query: str = Field(description="能力描述关键词，如 '公告'、'估值'、'回测'")


class SearchToolsTool(BaseTool):
    name = "search_tools"
    description = "按关键词检索可用工具（含尚未激活的懒加载工具）。"
    input_model = SearchToolsInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self, *, query: str) -> RawData:
        if self.registry is None:
            raise ValueError("当前会话未提供工具注册表")
        briefs = self.registry.search(query)
        if not briefs:
            text = "（未找到匹配工具）"
        else:
            text = "\n".join(
                f"- {b.name}（{b.group}，{'已激活' if self.registry.is_active(b.name) else '未激活'}）：{b.description}"
                for b in briefs
            )
        return RawData(kind="text", text=text, endpoint="meta:search_tools", params={"query": query})


class LoadToolInput(BaseModel):
    name: str = Field(description="要激活的工具名（来自 search_tools 结果）")


class LoadToolTool(BaseTool):
    name = "load_tool"
    description = "激活一个懒加载工具，使其在下一轮可以被调用。"
    input_model = LoadToolInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self, *, name: str) -> RawData:
        if self.registry is None:
            raise ValueError("当前会话未提供工具注册表")
        if self.registry.resolve(name) is None:
            raise ValueError(f"未找到工具：{name}")
        if not self.registry.is_active(name):
            self.registry.activate(name)
        if self.ctx is not None:
            self.ctx.activate_tool(name)
        # The schema appears in the *next* request, matching "the model may only
        # call tools whose schema it has been given".
        return RawData(
            kind="text",
            text=f"已激活工具 {name}，将在下一轮可调用。",
            endpoint="meta:load_tool",
            params={"name": name},
        )
