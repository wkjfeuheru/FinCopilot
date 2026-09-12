"""Integration: a real AgentLoop writes a report and the reviewer is triggered.

The unit tests cover each piece; this one proves they are wired together — the
loop injects the coordinator, ``write_report`` calls it, and the verdict reaches
the main transcript. The provider branches on the system prompt, so one scripted
object plays both the author and the reviewer.
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


class RoleBranchingProvider(Provider):
    """Author rounds for the main loop; a review answer for the sub-agent."""

    def __init__(self, author_rounds, review_text="### [中] 风险章节缺触发条件"):
        self.author_rounds = list(author_rounds)
        self.review_text = review_text
        self.systems: list[str] = []

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
        self.systems.append(system)
        # The checklist heading is unique to the reviewer's system prompt; the
        # main prompt also mentions "风险终审", so that would be ambiguous.
        if "风险核查清单" in system:
            # The reviewer is a pure-reading agent here: answer immediately.
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
    return Settings(
        # write_report is a WRITE tool with no whitelisted path, so it needs a
        # confirmation channel; auto mode is the non-interactive equivalent.
        permission=PermissionSettings(default_mode="auto"),
        data={"cache_dir": tmp_path / "cache"},
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
    # The report body cites this id, so the appendix and unsourced-number checks
    # have real provenance to work with — as they would in a live run.
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
    # The reviewer actually ran — one of the provider's calls carried its role.
    assert any("风险核查清单" in system for system in provider.systems)

    # Its comments reached the main transcript as a tool result.
    tool_contents = [
        json_content
        for message in loop.messages
        for _call_id, json_content in (message.tool_results or [])
    ]
    assert any("风险终审意见" in content for content in tool_contents)
    assert any("缺触发条件" in content for content in tool_contents)

    # The sidecar review file exists next to the report.
    reviews = list(Path(settings.paths.output_dir).glob("*.review.md"))
    assert reviews, "the review verdict must be written to disk"

    # And the spend is attributed to the sub-agent.
    snapshot = loop.stats.snapshot()
    assert snapshot.per_agent["risk"]["input_tokens"] == 9
    assert snapshot.per_agent["risk"]["output_tokens"] == 5
    assert snapshot.per_agent["risk"]["runs"] == 1
