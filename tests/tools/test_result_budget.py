"""工具结果预算的解析：单点、可声明、渲染与引擎一致。

覆盖的核心是那个真实缺陷：渲染侧对 ``detail="full"`` 乘放大倍数，引擎侧却不乘，
于是放宽渲染出的内容又被砍回原预算。修复的关键是两侧读取同一个解析器，
因此这里既测解析优先级，也测引擎实际放行的内容确实等于渲染所许的宽度。
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd

from finharness.config.settings import ContextSettings, Settings, ToolSettings
from finharness.data.access import DataAccess
from finharness.data.raw import RawData
from finharness.shared.budget import UNBOUNDED_RESULT_TOKENS, resolve_result_budget
from finharness.shared.declaration import Capability, tool
from finharness.tools.base import BaseTool

# -- 解析优先级 ---------------------------------------------------------------


def _tool(**attrs):
    """一个最小可用工具实例，仅用于解析预算。"""

    class Tool(BaseTool):
        name = "t"

    for key, value in attrs.items():
        setattr(Tool, key, value)
    return Tool(None)


def test_a_declared_tool_budget_beats_the_global_default():
    settings = Settings(context=ContextSettings(max_result_tokens=1000))

    assert resolve_result_budget(
        settings=settings, tool_name="t", tool=_tool(result_tokens=5000)
    ) == 5000


def test_an_operator_override_beats_the_tool_declaration():
    """运维覆盖优先于工具声明，与 timeout 的优先级一致。"""
    settings = Settings(
        context=ContextSettings(max_result_tokens=1000),
        tools=ToolSettings(result_token_overrides={"t": 777}),
    )

    assert resolve_result_budget(
        settings=settings, tool_name="t", tool=_tool(result_tokens=5000)
    ) == 777


def test_a_tool_without_a_declaration_falls_back_to_the_global_default():
    settings = Settings(context=ContextSettings(max_result_tokens=1234))

    assert resolve_result_budget(settings=settings, tool_name="t", tool=_tool()) == 1234


def test_no_settings_means_unbounded_not_zero():
    """拿不到配置时返回"不裁剪"，而不是 0——后者会把表格收缩到一列。"""
    assert resolve_result_budget(settings=None, tool_name="t", tool=_tool()) == (
        UNBOUNDED_RESULT_TOKENS
    )


def test_full_detail_multiplies_the_resolved_budget_once():
    settings = Settings(context=ContextSettings(max_result_tokens=1000))
    tool = _tool()

    base = resolve_result_budget(
        settings=settings, tool_name="t", tool=tool, detail="summary"
    )
    full = resolve_result_budget(
        settings=settings, tool_name="t", tool=tool, detail="full"
    )

    assert (base, full) == (1000, 4000)


# -- 引擎侧回归：full 渲染不被砍回 1× ------------------------------------------
#
# 这是修复前后的差别所在。用一台真实循环驱动一个声明了 detail="full" 的数据工具，
# 断言模型实际收到的内容长度接近渲染所许的 4×，而不是被压回 1×。


@tool(
    name="wide_tool",
    description="返回一张宽表",
    capability=Capability.FILE,
    data_tool=True,
)
class _FrameTool(BaseTool):
    async def _dispatch(self, **kwargs) -> RawData:
        df = pd.DataFrame({f"c{index}": ["值" * 20] for index in range(6)})
        return RawData(kind="df", df=df, endpoint="test:wide", params={})


def _run_loop_and_capture(settings: Settings, *, detail: str) -> str:
    from finharness.engine.loop import AgentLoop  # noqa: F401 - 触发循环注册
    from finharness.types import ToolUse
    from tests.engine.test_loop import (
        ScriptedProvider,
        StubRegistry,
        make_loop,
        text_round,
        tool_round,
    )

    tool = _FrameTool(_access(settings))
    provider = ScriptedProvider(
        [tool_round(ToolUse("call_1", "wide_tool", {"detail": detail})), text_round("ok")]
    )
    loop = make_loop(
        provider,
        registry=StubRegistry({"wide_tool": tool}),
        settings=settings,
    )
    asyncio.run(loop.run("要一张宽表"))
    return json.loads(loop.messages[2].tool_results[0][1])["content"]


def _access(settings: Settings) -> DataAccess:
    return DataAccess([], settings=settings)


def test_full_detail_content_is_not_clipped_back_to_the_summary_budget():
    settings = Settings(context=ContextSettings(max_result_tokens=60, trim_rows=20))

    summary = _run_loop_and_capture(settings, detail="summary")
    full = _run_loop_and_capture(settings, detail="full")

    # full 不只是一个稍微长一点的 summary：它拿到的是 4× 的预算。
    assert len(full) > len(summary) * 2
    # 且没有被引擎追加截断标记（渲染本就在预算内）。
    assert "已省略约" not in full
