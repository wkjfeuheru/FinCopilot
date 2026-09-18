"""上下文裁剪：内容在到达模型之前会被裁剪的部分（docs 3.6.1）。

裁剪只作用于 *上下文副本*。完整数据仍保留在缓存 parquet 中，因此不会有任何
丢失 —— 模型需要细节时可以再读回来。
"""

from __future__ import annotations

from typing import Any

from finharness.context.tokens import CHARS_PER_TOKEN

# 当调用方未提供时的默认描述长度预算。
DEFAULT_MAX_DESC_LEN = 60


def trim_schema(tool_schema: dict, *, max_desc_len: int = DEFAULT_MAX_DESC_LEN) -> dict:
    """就地缩短工具 schema 的描述，同时保持其结构不变。

    每个常驻工具的 schema 都会在每次请求中重新发送，因此描述长度是固定的
    每轮开销（docs 3.6.1）。裁剪落在字符边界上，并加上标记，让读者能看出
    它被缩短过。
    """
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return tool_schema
    description = function.get("description")
    if isinstance(description, str) and len(description) > max_desc_len:
        function["description"] = description[: max_desc_len - 1].rstrip() + "…"
    return tool_schema


def schema_tokens(schema: dict, *, counter: Any | None = None) -> int:
    """统计单个 schema 在请求中所占的 token 数。"""
    function = schema.get("function", {})
    text = str(function.get("description", "")) + str(function.get("parameters", ""))
    if counter is None:
        return int(len(text) / CHARS_PER_TOKEN)
    return counter.count(text).tokens
