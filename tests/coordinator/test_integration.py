"""集成测试：真实的 AgentLoop 写出报告并触发 reviewer。

单元测试覆盖了各个部分；本测试证明它们已串联在一起——loop 注入 coordinator，
``write_report`` 调用它，最终结论传入主 transcript。provider 依据 system prompt
分支，因此同一个脚本化对象可以同时扮演作者与 reviewer。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

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
from tests.conftest import settings_with_cache


class RoleBranchingProvider(Provider):
    """对主循环返回作者轮次；对 sub-agent 返回 review 答复。"""

    def __init__(self, author_rounds, review_text="### [中] 风险章节缺触发条件"):
        self.author_rounds = list(author_rounds)
        self.review_text = review_text
        self.systems: list[str] = []

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
        self.systems.append(system)
        # checklist 标题是 reviewer system prompt 独有的；
        # 主 prompt 也提到 "风险终审"，仅凭它会有歧义。
        if "风险核查清单" in system:
            # 这里 reviewer 是纯读取 agent：立即作答。
            yield StreamChunk(StreamEvent.TEXT_DELTA, self.review_text)
            yield StreamChunk(
                StreamEvent.MESSAGE_END, ModelUsage(input_tokens=9, output_tokens=5)
            )
            return
        script = self.author_rounds.pop(0) if self.author_rounds else []
        for chunk in script:
            yield chunk


def message_end(*tool_uses: ToolUse) -> StreamChunk:
    return StreamChunk(
        StreamEvent.MESSAGE_END, ModelUsage(input_tokens=3, output_tokens=1, tool_uses=list(tool_uses))
    )


def text_round(*parts: str) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in parts] + [message_end()]


def make_settings(tmp_path) -> Settings:
    return settings_with_cache(
        tmp_path,
        # write_report 是没有白名单路径的 WRITE tool，因此需要
        # 确认通道；auto 模式是非交互场景下的等价物。
        permission=PermissionSettings(default_mode="auto"),
        paths={"output_dir": tmp_path / "output"},
    )


def test_a_real_loop_reviews_the_report_it_wrote(tmp_path):
    settings = make_settings(tmp_path)
    provider = RoleBranchingProvider(
        [
            [
                message_end(
                    ToolUse(
                        "call_report",
                        "write_report",
                        {
                            "topic": "集成测试研报",
                            "core_view": ["观点 {cite:cit_000001}"],
                            "sections": [
                                {
                                    "heading": "章节",
                                    "body": "正文 {cite:cit_000001}",
                                    "cids": ["cit_000001"],
                                }
                            ],
                            "risks": ["风险一"],
                        },
                    )
                )
            ],
            text_round("报告已完成"),
        ]
    )
    data = DataAccess([], settings=settings)
    cite = CitationRegistry()
    # 报告正文引用了该 id，因此附录与无来源数字检查
    # 有真实的 provenance 可用——与实盘运行一致。
    cite.register(
        tool="get_quote", endpoint="fake", symbol="600519", params={},
        rows=1, cols=1, fingerprint="fp",
    )
    ctx = ResearchContext(cite=cite, settings=settings)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=system_prompt(),
        cite=cite,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
    )

    async def run():
        return await loop.run("请出一份集成测试研报")

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    # reviewer 确实运行了——provider 的某次调用携带了它的角色。
    assert any("风险核查清单" in system for system in provider.systems)

    # 它的意见作为 tool result 到达主 transcript。
    tool_contents = [
        json_content
        for message in loop.messages
        for _call_id, json_content in (message.tool_results or [])
    ]
    assert any("风险终审意见" in content for content in tool_contents)
    assert any("缺触发条件" in content for content in tool_contents)

    # 报告旁边存在 sidecar review 文件。
    reviews = list(Path(settings.paths.output_dir).glob("*.review.md"))
    assert reviews, "the review verdict must be written to disk"

    # 且开销被归因到 sub-agent。
    snapshot = loop.stats.snapshot()
    assert snapshot.per_agent["risk"]["input_tokens"] == 9
    assert snapshot.per_agent["risk"]["output_tokens"] == 5
    assert snapshot.per_agent["risk"]["runs"] == 1


def test_a_high_severity_finding_is_carried_into_the_session_state(tmp_path):
    """端到端：解析出的高严重度问题必须落进 ctx，并在后续轮次持续可见（docs 03.10.7）。

    这正是"严重问题必须让主 Agent 改"的结构性落点——不是提示词里的一句期望，
    而是一份挂在会话状态上、模型在之后每一轮都读得到的未结事项。
    """
    settings = make_settings(tmp_path)
    provider = RoleBranchingProvider(
        [
            [
                message_end(
                    ToolUse(
                        "call_report",
                        "write_report",
                        {
                            "topic": "集成测试研报",
                            "core_view": ["观点 {cite:cit_000001}"],
                            "sections": [
                                {
                                    "heading": "章节",
                                    "body": "正文 {cite:cit_000001}",
                                    "cids": ["cit_000001"],
                                }
                            ],
                            "risks": ["风险一"],
                        },
                    )
                )
            ],
            text_round("报告已完成"),
        ],
        review_text="### [高] 营收数字缺引用\n- 位置：财务摘要",
    )
    data = DataAccess([], settings=settings)
    cite = CitationRegistry()
    cite.register(
        tool="get_quote", endpoint="fake", symbol="600519", params={},
        rows=1, cols=1, fingerprint="fp",
    )
    ctx = ResearchContext(cite=cite, settings=settings)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=system_prompt(),
        cite=cite,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
    )

    outcome = asyncio.run(loop.run("请出一份集成测试研报"))

    assert outcome.succeeded is True, outcome.error
    # 未结事项按主题记账，且下一轮的状态块会把它带着走。
    assert "集成测试研报" in ctx.review_findings
    assert "[高] 营收数字缺引用" in ctx.review_findings["集成测试研报"]
    block = ctx.state_block()
    assert "未消解的风险终审问题" in block
    assert "不得声称已复核" in block
    # sidecar 仍然落盘（取证与闩锁），但它不是交付物。
    assert list(Path(settings.paths.output_dir).glob("*.review.md"))
