"""spawn_agent：输入处理、渲染与 coordinator 契约。

coordinator 的 fan-out 行为在 tests/coordinator/test_spawn.py 中测试；
这里由一个 stub coordinator 提供预设的 SubAgentResult，因此被测的是
该 tool 自身的行为——校验、渲染、失败上报。
"""

from __future__ import annotations

import asyncio

from finharness.coordinator import SubAgentResult
from finharness.tools.base import PermissionLevel, ToolGroup
from finharness.tools.meta.spawn import SpawnAgentTool
from finharness.tools.registry import DEFAULT_LAZY_TOOLS


class StubCoordinator:
    def __init__(self, results: list[SubAgentResult]):
        self.results = results
        self.calls: list[dict] = []

    async def spawn(self, *, tasks, focus, context=None):
        self.calls.append({"tasks": tasks, "focus": focus, "context": context})
        return self.results


def make_tool(results: list[SubAgentResult]) -> tuple[SpawnAgentTool, StubCoordinator]:
    tool = SpawnAgentTool(data=None)  # type: ignore[arg-type]
    coordinator = StubCoordinator(results)
    tool.coordinator = coordinator
    return tool, coordinator


def run(coro):
    return asyncio.run(coro)


def test_results_are_rendered_one_block_per_task():
    tool, _ = make_tool(
        [
            SubAgentResult(focus="general", task="摘要甲文件", summary="甲的核心结论"),
            SubAgentResult(focus="general", task="摘要乙文件", summary="乙的核心结论"),
        ]
    )

    result = run(tool.run(tasks=["摘要甲文件", "摘要乙文件"]))

    assert result.ok is True, result.error
    assert "甲的核心结论" in result.content
    assert "乙的核心结论" in result.content
    assert "子任务 1" in result.content and "子任务 2" in result.content
    # 任务被回显，因此某个结论不会被误读为另一个任务的结论。
    assert "摘要甲文件" in result.content


def test_a_failed_task_is_stated_as_a_failure_not_left_blank():
    tool, _ = make_tool(
        [
            SubAgentResult(focus="general", task="甲", summary="好的"),
            SubAgentResult(focus="general", task="乙", summary="", ok=False, error="provider boom"),
        ]
    )

    result = run(tool.run(tasks=["甲", "乙"]))

    assert result.ok is True
    assert "成功 1 个" in result.content
    assert "未能完成" in result.content
    assert "provider boom" in result.content


def test_citations_are_surfaced_as_placeholders():
    """worker 可以引用它所读到的内容；这些 cid 必须传递到主 agent。"""
    tool, _ = make_tool(
        [
            SubAgentResult(
                focus="general", task="甲", summary="结论", citations=["cit_000007"]
            )
        ]
    )

    result = run(tool.run(tasks=["甲"]))

    assert "{cite:cit_000007}" in result.content


def test_the_general_focus_is_requested():
    tool, coordinator = make_tool([SubAgentResult(focus="general", task="甲", summary="x")])

    run(tool.run(tasks=["甲"], context="共享背景"))

    assert coordinator.calls[0]["focus"] == "general"
    assert coordinator.calls[0]["context"] == "共享背景"


def test_an_unwired_coordinator_fails_clearly():
    tool = SpawnAgentTool(data=None)  # type: ignore[arg-type]
    # 未注入 coordinator（例如 loop 没有可提供的）。
    result = run(tool.run(tasks=["甲"]))

    assert result.ok is False
    assert "协调器" in result.error


def test_an_empty_task_list_is_a_validation_error():
    tool, _ = make_tool([])

    result = run(tool.run(tasks=[]))

    assert result.ok is False
    assert "tasks" in result.error


def test_the_tool_is_lazy_read_only_meta():
    assert SpawnAgentTool.permission is PermissionLevel.READ
    assert SpawnAgentTool.group is ToolGroup.META
    assert SpawnAgentTool.needs_coordinator is True
    assert "spawn_agent" in DEFAULT_LAZY_TOOLS
