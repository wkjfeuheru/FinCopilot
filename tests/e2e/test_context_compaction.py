"""M4 acceptance: the window stays bounded on a long real session.

WBS M4 asks for "10 turns without blowing the window". The claim under test is
therefore about the *window*, not about whether an open-ended research task
finishes within a turn budget — that depends on the model re-fetching data,
which is L2's job (deferred). So this asserts:

* compaction engages once the transcript grows,
* it brings the window back under the budget, and
* the final window is still inside the model's limit.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from finharness.config.settings import ContextSettings, PermissionSettings, Settings
from finharness.context.session import ResearchContext
from finharness.context.tokens import TokenCounter
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.permissions.gate import PermissionGate
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke

# Exercise the shipped prompt so the asset itself is under test.
SYSTEM_PROMPT = system_prompt()

# Deliberately small so a normal multi-step task crosses it and must compact.
WINDOW_TOKENS = 12000


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def _build(tmp_path):
    import httpx

    settings = Settings(
        permission=PermissionSettings(default_mode="auto"),
        context=ContextSettings(
            context_window_tokens=WINDOW_TOKENS, compaction_ratio=0.6, max_turns=30
        ),
        data={"cache_dir": tmp_path / "cache"},
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
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=SYSTEM_PROMPT,
        cite=citations,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        counter=TokenCounter(cache_dir=tmp_path / "cache" / "tiktoken"),
    )
    return loop, settings


def test_long_session_keeps_the_window_bounded(tmp_path):
    """A data-heavy session must compact and stay inside the window."""
    loop, settings = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(
            loop.run(
                "请研究贵州茅台(600519)：先看最新行情与近一年股价走势，"
                "再看财务指标与杜邦分解，再看同业估值对比。"
            ),
            600,
        )

    outcome = asyncio.run(run())

    # A transient provider failure means the window was never really exercised;
    # report that honestly rather than passing vacuously.
    if outcome.reason == "provider_error":
        pytest.skip(f"provider error, window not exercised: {outcome.error}")

    # The run may or may not finish within the turn budget, but it must never
    # have blown the window — that is what M4 delivers.
    hard_limit = WINDOW_TOKENS
    assert loop.compactions, "the session should have grown enough to compact"
    assert loop._window_tokens() < hard_limit, (
        f"window {loop._window_tokens()} exceeded the {hard_limit} limit"
    )
    for result in loop.compactions:
        assert result.after_tokens < hard_limit, "compaction must leave room to continue"
        assert result.compacted is True
    # Either it converged, or it ran out of turns having kept the window small.
    assert outcome.succeeded is True or outcome.reason == "max_turns_exhausted"


def test_compaction_engages_on_a_multi_step_task(tmp_path):
    """A task that crosses the small window must actually trigger compaction."""
    loop, _ = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(
            loop.run(
                "对比 600519 与 000858 的行情、财务指标与估值，并给出风险提示。"
            ),
            600,
        )

    asyncio.run(run())

    assert loop.compactions, "a multi-step data task should exceed the small window"
    first = loop.compactions[0]
    # The reduction must be real, not incidental.
    assert first.after_tokens < first.before_tokens
    assert first.removed > 0
