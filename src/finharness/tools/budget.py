"""工具结果预算的单点解析（docs 03.3.3 · 03.6.1）。

一次工具结果在写入对话记录之前会被两个地方裁剪：工具自身渲染时（``trim_dataframe``），
以及引擎编码结果时（``AgentLoop._truncate``）。两处过去各自读取
``context.max_result_tokens``，于是出现了真实的缺陷：渲染侧对 ``detail="full"`` 乘了放大
倍数，引擎侧没有，因此放宽渲染出的内容又被砍回原预算——"再多给点"这个逃生口有一半
是失效的。

把解析收敛到本模块后，两侧读的是同一个数字，不可能再分叉。优先级与超时一致
（``loop.py`` 的 timeout 解析），因此两个预算的调优方式是对称的：

1. 运维覆盖 ``tools.result_token_overrides[name]``
2. 工具自身声明 ``BaseTool.result_tokens``
3. 全局默认 ``context.max_result_tokens``
"""

from __future__ import annotations

from typing import Any

# ``detail="full"`` 时渲染预算放大的倍数。它是一个有界倍数而非无界：目的是让调用方
# 已在使用的工具就能回答"再多给点"，而不是另设一个工具重读已取数据。即使是 ``full``
# 也仍在上下文的 token 约束之内。
FULL_DETAIL_MULTIPLIER = 4

# 拿不到配置时使用的"不裁剪"预算。它必须是真正意义上的无限（而不是 0）：裁剪循环
# 的判据是 ``count(candidate) <= budget``，若返回 0 会把表格一路收缩到只剩一列——这与
# "未知预算就原样呈现"的意图正好相反。
UNBOUNDED_RESULT_TOKENS = 10**9


def resolve_result_budget(
    *,
    settings: Any | None,
    tool_name: str,
    tool: Any | None = None,
    detail: str = "summary",
) -> int:
    """解析单条工具结果可用的 token 预算。

    ``detail="full"`` 的放大只在**此处**应用一次，因此渲染侧与引擎侧得到相同的上限。
    """
    declared = getattr(tool, "result_tokens", None)
    base = _base_budget(settings, tool_name=tool_name, declared=declared)
    if detail == "full":
        return base * FULL_DETAIL_MULTIPLIER
    return base


def _base_budget(settings: Any | None, *, tool_name: str, declared: Any) -> int:
    """按 覆盖 → 声明 → 全局 的顺序取基准预算（未放大）。"""
    overrides = getattr(getattr(settings, "tools", None), "result_token_overrides", None)
    if overrides:
        override = overrides.get(tool_name)
        if override:
            return int(override)
    if declared:
        return int(declared)
    context = getattr(settings, "context", None)
    default = getattr(context, "max_result_tokens", None)
    if default is None:
        return UNBOUNDED_RESULT_TOKENS
    return int(default)
