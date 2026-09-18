"""集成测试：真实的 AgentLoop 通过 spawn_agent 派发 sub-agent（文档 03.10）。

单元测试证明 coordinator 会 fan-out、tool 会渲染；本测试证明它们通过真实 loop
串联在一起——loop 把 coordinator 注入 ``spawn_agent``，fan-out 执行，结论到达主
transcript，而 sub-agent 的工作不会混入其中。

同一个 provider 同时扮演主 agent 与各 worker，依据 system prompt 分支，
从而让单个脚本化对象服务所有上下文。
"""

from __future__ import annotations

import asyncio

from finharness.config.settings import PermissionSettings, Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.permissions.gate import PermissionGate
from finharness.provider.base import Provider
from finharness.tools.registry import ToolRegistry
from finharness.types import ModelUsage, StreamChunk, StreamEvent, ToolUse


class DispatchProvider(Provider):
    """规划一次 spawn_agent 调用，然后用各自的任务答复每个 worker。"""

    def __init__(self, tasks: list[str]):
        self.tasks = tasks
        self.worker_systems: list[str] = []
        self.worker_inputs: list[str] = []
        self.main_calls = 0

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
        # worker 角色 prompt 是独特的：它点明了隔离的 sub-agent。
        if "独立执行者" in system:
            self.worker_systems.append(system)
            text = "\n".join(
                getattr(m, "content", "") or ""
                for m in messages
                if getattr(m, "role", None) == "user"
            )
            self.worker_inputs.append(text)
            # 回显任务，使结论可追溯到对应的 worker。
            yield StreamChunk(StreamEvent.TEXT_DELTA, f"[摘要]{text.strip()[:24]}")
            yield StreamChunk(
                StreamEvent.MESSAGE_END, ModelUsage(input_tokens=7, output_tokens=3)
            )
            return

        self.main_calls += 1
        if self.main_calls == 1:
            yield StreamChunk(
                StreamEvent.MESSAGE_END,
                ModelUsage(
                    input_tokens=3,
                    output_tokens=1,
                    tool_uses=[ToolUse("c1", "spawn_agent", {"tasks": self.tasks})],
                ),
            )
            return
        yield StreamChunk(StreamEvent.TEXT_DELTA, "汇总完成")
        yield StreamChunk(StreamEvent.MESSAGE_END, ModelUsage(input_tokens=3, output_tokens=1))


def make_settings(tmp_path) -> Settings:
    return Settings(
        permission=PermissionSettings(default_mode="auto"),
        data={"cache_dir": tmp_path / "cache"},
        paths={"output_dir": tmp_path / "output"},
    )


def build_loop(tmp_path, provider):
    settings = make_settings(tmp_path)
    data = DataAccess([], settings=settings)
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    registry = ToolRegistry(data, ctx=ctx, settings=settings)
    # spawn_agent 是按需工具。直接调用它已经不再被拒绝（就地激活），
    # 但这里仍显式激活一次，使被测脚本的第一步与"已发现它"的常态一致。
    registry.activate("spawn_agent")
    loop = AgentLoop(
        provider=provider,
        registry=registry,
        settings=settings,
        system=system_prompt(),
        cite=cite,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
    )
    return loop, settings


def tool_contents(loop: AgentLoop) -> list[str]:
    return [
        content
        for message in loop.messages
        for _call_id, content in (message.tool_results or [])
    ]


def test_a_real_loop_fans_out_tasks_and_collects_conclusions(tmp_path):
    tasks = ["摘要研报甲", "摘要研报乙", "摘要研报丙"]
    provider = DispatchProvider(tasks)
    loop, _ = build_loop(tmp_path, provider)

    outcome = asyncio.run(loop.run("把这三份研报分别摘要"))

    assert outcome.succeeded is True, outcome.error

    # 每个任务都在各自的 sub-agent 上下文中运行。
    assert len(provider.worker_systems) == 3
    assert len(provider.worker_inputs) == 3
    # 且每个 worker 只看到自己的任务。
    for task, seen in zip(tasks, provider.worker_inputs):
        assert task in seen
        assert sum(other in seen for other in tasks) == 1

    # 每个结论都到达主 transcript，并标注了它属于哪个任务。
    contents = "\n".join(tool_contents(loop))
    assert "汇总完成" in contents or "摘要研报甲" in contents
    assert "子任务 1" in contents and "子任务 3" in contents

    # 成本归因到 general focus，每个任务一次运行。
    snapshot = loop.stats.snapshot()
    assert snapshot.per_agent["general"]["runs"] == 3
    assert snapshot.per_agent["general"]["input_tokens"] == 21  # 3 x 7
    # 各 worker 自身的 tool 使用不得显示为主 agent 的 tools。
    assert "read_file" not in snapshot.per_tool


def test_the_worker_transcript_does_not_enter_the_main_memory(tmp_path):
    """隔离正是关键：只返回结论，不返回运行过程。"""
    provider = DispatchProvider(["摘要研报甲", "摘要研报乙"])
    loop, _ = build_loop(tmp_path, provider)

    asyncio.run(loop.run("摘要这两份"))

    # 主 transcript 保存了 spawn_agent 的结果，但没有任何 worker 角色文本
    # 泄漏进 user/assistant 消息。
    for message in loop.messages:
        content = getattr(message, "content", None) or ""
        assert "独立执行者" not in content
    # worker prompt 只作为 sub-agent 的 system prompt 出现过。
    assert all("独立执行者" in system for system in provider.worker_systems)
