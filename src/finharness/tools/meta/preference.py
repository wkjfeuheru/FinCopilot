"""remember_preference：记录一条对该用户所有对话共享的偏好。

偏好以*用户*（而非对话）为作用域——“这位用户喜欢怎样的报告”处处适用——因此本
工具写入按用户划分的 ``notes`` 表，而非对话自身的记忆。

声明为 READ，使其无需确认提示即被允许：记录用户刚刚表达的偏好正是其要求，而每次提及
都弹确认会让该功能无法使用。它从下一轮起生效，因为记忆在提示构建之前就已组装完成。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class RememberPreferenceInput(BaseModel):
    key: str = Field(description="偏好名，简短稳定，如 report_style、preferred_period")
    value: str = Field(description="偏好取值，如「简洁，少用表格」")


class RememberPreferenceTool(BaseTool):
    name = "remember_preference"
    description = "记住用户在本次或以后对话中都适用的偏好（跨对话共享）。"
    input_model = RememberPreferenceInput
    permission = PermissionLevel.READ
    group = ToolGroup.META
    timeout = 10

    async def _dispatch(self, *, key: str, value: str) -> RawData:
        """把偏好写入全局存储；无存储时退化为仅当前会话生效。"""
        store = getattr(self.ctx, "store", None) if self.ctx is not None else None
        if store is None:
            # 退回到会话视图，使调用者仍能看到效果。
            if self.ctx is not None:
                self.ctx.notes[key] = value
            return RawData(
                kind="text",
                text=f"已记录偏好（仅当前会话有效）：{key}={value}",
                endpoint="memory:remember_preference",
                params={"key": key},
            )
        user_id = getattr(self.ctx, "user_id", "") if self.ctx is not None else ""
        store.set_note(key, value, user_id=user_id, kind="preference")
        self.ctx.notes[key] = value
        return RawData(
            kind="text",
            text=f"已记住偏好（对所有对话生效）：{key}={value}",
            endpoint="memory:remember_preference",
            params={"key": key},
        )
