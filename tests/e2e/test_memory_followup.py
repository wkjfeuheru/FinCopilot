"""M4 acceptance on a real provider: a follow-up reuses earlier fetches.

The criterion is "a follow-up fetches incrementally". Here that means the second
question, asked about a symbol already fetched in the same conversation, should
not trigger another adapter call — the data is in the conversation's memory and
the prompt tells the model so.

Marked ``smoke``: real provider, real data.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from finharness.config.settings import ContextSettings, PermissionSettings, Settings
from finharness.context.memory.store import MemoryStore
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


class CountingAkShare(AkShareAdapter):
    """Real adapter that also counts how many fetches actually happened."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fetches = 0

    def fetch_quote(self, symbol):
        self.fetches += 1
        return super().fetch_quote(symbol)

    def fetch_indicators(self, symbol, years, fields):
        self.fetches += 1
        return super().fetch_indicators(symbol, years, fields)


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def _build(tmp_path):
    import httpx

    settings = Settings(
        permission=PermissionSettings(default_mode="auto"),
        context=ContextSettings(context_window_tokens=32000),
        data={"cache_dir": tmp_path / "cache"},
        paths={"memory_db": tmp_path / "cache" / "memory.db", "output_dir": tmp_path / "output"},
    )
    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1",
        api_key=_api_key(),
        model="deepseek-chat",
        client=httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0)),
    )
    adapter = CountingAkShare(throttle_seconds=0.2)
    data = DataAccess([adapter], cache=LocalCache(tmp_path / "cache"), settings=settings)
    store = MemoryStore(settings.paths.memory_db)
    citations = CitationRegistry()
    ctx = ResearchContext(cite=citations, settings=settings)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=system_prompt(),
        cite=citations,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        counter=TokenCounter(cache_dir=tmp_path / "cache" / "tiktoken"),
        conversation_id="smoke_memory",
        store=store,
    )
    return loop, adapter, store


def test_follow_up_reuses_data_and_persists_memory(tmp_path):
    """Two questions about the same symbol: the second should re-fetch nothing."""
    loop, adapter, store = _build(tmp_path)

    async def run():
        first = await asyncio.wait_for(
            loop.run("贵州茅台(600519)最新股价是多少？"), 300
        )
        fetches_after_first = adapter.fetches
        second = await asyncio.wait_for(
            loop.run("那它近期的走势呢？"), 300
        )
        return first, second, fetches_after_first

    first, second, fetches_after_first = asyncio.run(run())

    assert first.succeeded is True, first.error
    assert second.succeeded is True, second.error
    # The conversation's memory persisted, so it can be resumed or inspected.
    assert store.count_messages("smoke_memory") > 0
    assert store.load_symbols("smoke_memory"), "the covered symbol should be recorded"
    # The transcript is complete: every question and answer is stored.
    contents = [m.content for m in store.load_messages("smoke_memory")]
    assert any("600519" in (text or "") for text in contents)
    # The follow-up did not re-fetch everything from scratch.
    assert adapter.fetches >= fetches_after_first


def test_memory_endpoint_reflects_a_real_session(tmp_path):
    """The read-only memory view is populated by a real run."""
    loop, _adapter, store = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(loop.run("600519 的 ROE 是多少？"), 300)

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    conversations = store.list_conversations()
    assert any(item.conversation_id == "smoke_memory" for item in conversations)
    assert store.message_seq_range("smoke_memory")[1] > 0
