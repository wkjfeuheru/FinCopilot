"""M3 end-to-end: a natural-language request produces a charted report with an appendix.

Hits a real provider and real market data, so it is marked ``smoke``.
Requires a CJK font; skips cleanly where none is installed.
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
from finharness.report.charting import FontUnavailableError, resolve_cjk_font
from finharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke

# Exercise the shipped prompt so the asset itself is under test.
SYSTEM_PROMPT = system_prompt()

# M5 budget envelope (docs §10). Measured on a from-scratch report request
# against deepseek-chat: 102s wall, 300,319 total tokens (28 tool calls, and the
# report revised twice after review). These ceilings sit above the measurement
# with headroom, so they catch a gross regression — an unbounded revise loop, a
# lost cache — without failing on ordinary model-to-model variance.
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
    """One request should yield both artefact forms, with citations and a chart."""
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
    # The charted page and the provenance appendix are the M3 acceptance points.
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
    # Data actually fetched during the run is what the appendix is built from.
    assert ctx.symbols, "the session should have recorded covered symbols"


def test_report_generation_triggers_the_risk_review_sub_agent(tmp_path):
    """M5 acceptance: a report request earns an independent risk review (docs 03.10).

    The demo claim is "writing a report runs a reviewer that did not write it".
    This asserts the observable outcome — a review file with comments and a
    per-agent token entry — rather than the mechanism, so it stays honest if the
    internals move.

    It also enforces the M5 budget envelope: the same run must finish inside the
    time and token ceilings. That is the point of putting the measurement here —
    a cost regression and a review regression should both fail this one test.
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
    # Either real comments or an explicit clean bill; an empty file is a bug.
    assert "风险终审意见" in comments
    assert comments.split("风险终审意见", 1)[1].strip()

    # Review spend is attributed to the session under its own focus.
    snapshot = loop.stats.snapshot()
    assert "risk" in snapshot.per_agent
    assert snapshot.per_agent["risk"]["runs"] >= 1
    assert snapshot.per_agent["risk"]["input_tokens"] > 0

    # M5 budget envelope.
    total_tokens = snapshot.input_tokens + snapshot.output_tokens
    assert wall_s < REPORT_WALL_CEILING_S, (
        f"report took {wall_s:.0f}s, over the {REPORT_WALL_CEILING_S:.0f}s ceiling"
    )
    assert total_tokens < REPORT_TOKEN_CEILING, (
        f"report spent {total_tokens} tokens, over the {REPORT_TOKEN_CEILING} ceiling"
    )
