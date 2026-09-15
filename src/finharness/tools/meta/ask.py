"""ask_user：暂停当前轮次并向用户提问（docs 03.7.1）。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class AskUserInput(BaseModel):
    question: str = Field(description="要问用户的问题，需具体、可回答")
    options: list[str] = Field(
        default_factory=list, description="可选项；留空表示自由文本回答"
    )


class AskUserTool(BaseTool):
    name = "ask_user"
    description = "当关键信息缺失且无法从数据推断时，向用户提问以澄清需求。"
    input_model = AskUserInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 130
    needs_interactive = True

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
