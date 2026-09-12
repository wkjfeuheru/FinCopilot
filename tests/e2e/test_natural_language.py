"""End-to-end acceptance: natural language -> agent picks a tool -> grounded reply.

These tests hit a real LLM provider and real market-data sources, so they are
marked ``smoke`` and skipped unless credentials and network are available.
Run with: ``uv run pytest -m smoke``; the default suite excludes them.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
SYSTEM_PROMPT = (
    "You are FinHarness, a financial research copilot. "
    "Use the provided tools to fetch factual market data before answering."
)


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def _build_loop(tmp_path) -> AgentLoop:
    import httpx

    provider = OpenAICompatProvider(
        base_url=DEEPSEEK_BASE_URL,
        api_key=_api_key(),
        model=DEEPSEEK_MODEL,
        client=httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0)),
    )
    data = DataAccess(
        [AkShareAdapter(throttle_seconds=0.2)],
        cache=LocalCache(tmp_path / "cache"),
        settings=Settings(),
    )
    return AgentLoop(
        provider=provider,
        registry=ToolRegistry(data),
        settings=Settings(),
        system=SYSTEM_PROMPT,
        cite=CitationRegistry(),
    )


def test_natural_language_question_routes_to_a_tool_and_returns_grounded_data(tmp_path):
    """A price question must reach a market-data tool and yield a real number."""
    loop = _build_loop(tmp_path)

    async def run():
        return await asyncio.wait_for(loop.run("贵州茅台(600519)最新股价是多少？只回答价格。"), 180)

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    assert outcome.tool_calls >= 1, "the model should have called a data tool"
    # A grounded answer contains digits from the fetched frame.
    assert any(ch.isdigit() for ch in outcome.answer)
    # Provenance was recorded for the data actually used.
    assert outcome.citations, "a data-backed answer must register citations"


def test_natural_language_kline_question_uses_history_tool(tmp_path):
    loop = _build_loop(tmp_path)

    async def run():
        return await asyncio.wait_for(loop.run("帮我看看贵州茅台(600519)近一年的股价走势。"), 180)

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    assert outcome.tool_calls >= 1
    assert outcome.citations
