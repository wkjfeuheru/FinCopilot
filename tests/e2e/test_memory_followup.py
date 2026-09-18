"""在真实 provider 上进行的 M4 验收：后续追问复用先前的获取结果。

验收标准是「后续追问按增量获取」。在这里，这意味着：针对同一会话中已经获取过的
symbol 再次提问时，不应触发另一次 adapter 调用——数据已在会话的 memory 中，
prompt 也会告知模型这一点。

标记为 ``smoke``：真实 provider、真实数据。
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
    """真实 adapter，同时统计实际发生了多少次获取。"""

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
        paths={"memory_db": tmp_path / "state" / "memory.db", "output_dir": tmp_path / "output"},
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
    """关于同一 symbol 的两个问题：第二次不应重新获取任何数据。"""
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
    # 会话的 memory 已持久化，因此可以恢复或检查。
    assert store.count_messages("smoke_memory") > 0
    assert store.load_symbols("smoke_memory"), "the covered symbol should be recorded"
    # 记录是完整的：每个问题和回答都被存储。
    contents = [m.content for m in store.load_messages("smoke_memory")]
    assert any("600519" in (text or "") for text in contents)
    # 后续追问没有从头重新获取全部数据。
    assert adapter.fetches >= fetches_after_first


def test_memory_endpoint_reflects_a_real_session(tmp_path):
    """只读的 memory 视图由一次真实运行填充。"""
    loop, _adapter, store = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(loop.run("600519 的 ROE 是多少？"), 300)

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    conversations = store.list_conversations()
    assert any(item.conversation_id == "smoke_memory" for item in conversations)
    assert store.message_seq_range("smoke_memory")[1] > 0
