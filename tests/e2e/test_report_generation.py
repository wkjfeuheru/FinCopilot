"""M3 end-to-end: a natural-language request produces a charted report with an appendix.

Hits a real provider and real market data, so it is marked ``smoke``.
Requires a CJK font; skips cleanly where none is installed.
"""

from __future__ import annotations

import asyncio
import os
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
