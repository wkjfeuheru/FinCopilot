"""remember_preference: record a user preference shared by all conversations.

Preferences are the one memory surface that is *not* conversation-scoped — "how
this user likes reports" applies everywhere — so this tool writes to the global
``notes`` table rather than the conversation's own memory.

Declared READ so it is allowed without a confirmation prompt: recording a stated
preference is what the user just asked for, and prompting on every mention would
make the feature unusable. It takes effect from the next turn, because memory is
assembled before the prompt is built.
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
        store = getattr(self.ctx, "store", None) if self.ctx is not None else None
        if store is None:
            # Fall back to the session view so the caller still sees the effect.
            if self.ctx is not None:
                self.ctx.notes[key] = value
            return RawData(
                kind="text",
                text=f"已记录偏好（仅当前会话有效）：{key}={value}",
                endpoint="memory:remember_preference",
                params={"key": key},
            )
        store.set_note(key, value, kind="preference")
        self.ctx.notes[key] = value
        return RawData(
            kind="text",
            text=f"已记住偏好（对所有对话生效）：{key}={value}",
            endpoint="memory:remember_preference",
            params={"key": key},
        )
