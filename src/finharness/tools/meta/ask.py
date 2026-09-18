"""ask_user：暂停当前轮次并向用户提问（docs 03.7.1）。"""

from __future__ import annotations

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool


@tool(
    name="ask_user",
    description="当关键信息缺失且无法从数据推断时，向用户提问以澄清需求。",
    capability=Capability.META,
    group=ToolGroup.META,
    timeout=130,
    needs_interactive=True,
)
class AskUserTool(BaseTool):
    @param("question", desc="要问用户的问题，需具体、可回答")
    @param("options", desc="可选项；留空表示自由文本回答")
    async def _dispatch(self, *, question: str, options: list[str] | None = None) -> RawData:
        """经由注入的交互回调向用户提问；超时未答时返回提示模型继续的文本。"""
        if self.interactive is None:
            raise ValueError("当前环境不支持交互提问")
        answer = await self.interactive("question", question, list(options or []))
        if answer is None:
            text = "用户未在时限内回答该问题，请基于已有信息继续或说明不确定。"
        else:
            text = f"用户回答：{answer}"
        return RawData(
            kind="text",
            text=text,
            endpoint="meta:ask_user",
            params={"question": question, "answered": answer is not None},
        )
