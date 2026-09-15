"""M3 端到端：一个自然语言请求会生成带图表的 report 和附录。

会访问真实 provider 和真实市场数据，因此标记为 ``smoke``。
需要 CJK 字体；未安装时会被干净地跳过。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from finharness.config.settings import PermissionSettings, Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import HookChain
from finharness.permissions.gate import PermissionGate
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.tools.fin.charting import FontUnavailableError, resolve_cjk_font
from finharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke

# 使用随包发布的 prompt，以便该资产本身也处于测试之下。
SYSTEM_PROMPT = system_prompt()

# M5 预算上限（文档 §10）。基于一次从零开始的 report 请求测得
# （针对 deepseek-chat）：墙钟 102s，总计 300,319 tokens（28 次 tool call，
# report 经 review 后修订了两次）。这些上限在测量值之上留有余量，
# 因此能捕获严重回归——无界的 revise 循环、缓存丢失——而不会因
# 模型之间的正常差异而失败。
REPORT_WALL_CEILING_S = 240.0
REPORT_TOKEN_CEILING = 600_000


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def _require_font() -> None:
    try:
        resolve_cjk_font()
    except FontUnavailableError as exc:
        pytest.skip(f"no CJK font available: {exc}")


def _build(tmp_path):
    import httpx

    settings = Settings(
        permission=PermissionSettings(default_mode="auto"),
        data={"cache_dir": tmp_path / "cache"},
        audit={"log_path": tmp_path / "audit.jsonl"},
        paths={"output_dir": tmp_path / "output"},
    )
    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1",
        api_key=_api_key(),
        model="deepseek-chat",
        client=httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0)),
    )
    data = DataAccess(
        [AkShareAdapter(throttle_seconds=0.2)], cache=LocalCache(tmp_path / "cache"), settings=settings
    )
    citations = CitationRegistry()
    ctx = ResearchContext(cite=citations, settings=settings)
    writer = AuditLogWriter(settings.audit.log_path)
    audit = AuditHook(writer, session_id="m3")
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=SYSTEM_PROMPT,
        cite=citations,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        hooks=HookChain([audit]),
    )
    loop.audit = audit
    return loop, ctx, settings


def test_report_request_produces_a_charted_docx(tmp_path):
    """一次请求应产出两种形态的交付物，并带有 citations 和一张 chart。"""
    _require_font()
    loop, _, settings = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(
            loop.run(
                "请基于本次会话研究贵州茅台(600519)，给我出一份带图表的估值分析研报。"
            ),
            600,
        )

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    output_dir = Path(settings.paths.output_dir)
    markdowns = list(output_dir.glob("*.md"))
    docx_files = list(output_dir.glob("*.docx"))

    assert markdowns, "a markdown report must be produced"
    assert docx_files, "a docx must be produced alongside the markdown"
    # 带图表的页面和来源附录是 M3 的验收要点。
    assert list((output_dir / "charts").glob("*.png")), "the report should embed a chart"
    body = markdowns[0].read_text(encoding="utf-8")
    assert "附录" in body
    assert outcome.citations, "report conclusions must carry citations"


def test_report_request_registers_citations_for_the_appendix(tmp_path):
    _require_font()
    loop, ctx, settings = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(
            loop.run("帮我出一份 600519 的研报，先取数再成稿。"), 600
        )

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    # 附录正是基于运行期间实际获取的数据构建的。
    assert ctx.symbols, "the session should have recorded covered symbols"


def test_report_generation_triggers_the_risk_review_sub_agent(tmp_path):
    """M5 验收：一个 report 请求会触发独立的 risk review（文档 03.10）。

    demo 的主张是「写 report 会运行一个并非由它自己撰写的 reviewer」。
    这里断言的是可观察的结果——一个带有意见的 review 文件，以及一条按 agent 归集的
    token 记录——而不是其内部机制，这样即使内部实现变动也依然成立。

    它还强制 M5 预算上限：同一次运行必须在时间和 token 上限内完成。
    这正是把测量放在此处的原因——成本回归和 review 回归都应在这一个测试中失败。
    """
    _require_font()
    loop, _, settings = _build(tmp_path)

    assert loop.coordinator is not None, "the loop should have a risk reviewer"

    async def run():
        return await asyncio.wait_for(
            loop.run("请研究贵州茅台(600519)的基本面并出一份研报，务必包含风险提示章节。"),
            900,
        )

    started = time.monotonic()
    outcome = asyncio.run(run())
    wall_s = time.monotonic() - started

    assert outcome.succeeded is True, outcome.error
    output_dir = Path(settings.paths.output_dir)
    reviews = list(output_dir.glob("*.review.md"))
    assert reviews, "writing a report must produce a risk review file"
    comments = reviews[0].read_text(encoding="utf-8")
    # 要么是真实的意见，要么是明确的「无问题」结论；空文件是一个 bug。
    assert "风险终审意见" in comments
    assert comments.split("风险终审意见", 1)[1].strip()

    # review 的花费会在其专门的 focus 下归集到本会话。
    snapshot = loop.stats.snapshot()
    assert "risk" in snapshot.per_agent
    assert snapshot.per_agent["risk"]["runs"] >= 1
    assert snapshot.per_agent["risk"]["input_tokens"] > 0

    # M5 预算上限。
    total_tokens = snapshot.input_tokens + snapshot.output_tokens
    assert wall_s < REPORT_WALL_CEILING_S, (
        f"report took {wall_s:.0f}s, over the {REPORT_WALL_CEILING_S:.0f}s ceiling"
    )
    assert total_tokens < REPORT_TOKEN_CEILING, (
        f"report spent {total_tokens} tokens, over the {REPORT_TOKEN_CEILING} ceiling"
    )
