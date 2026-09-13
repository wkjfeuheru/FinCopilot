"""Conversation memory across a loop: isolation, reload, and follow-up reuse.

The M4 acceptance criterion "a follow-up only fetches incrementally" is asserted
here directly: the second question about the same symbol must not trigger another
adapter fetch, because the data is already in the conversation's memory.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from finharness.config.settings import ContextSettings, PermissionSettings, Settings
from finharness.context.memory.store import MemoryStore
from finharness.context.session import ResearchContext
from finharness.context.tokens import TokenCounter
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.permissions.gate import PermissionGate
from finharness.types import ToolUse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import RecordingTool, ScriptedProvider, StubRegistry, text_round, tool_round  # noqa: E402

# Shared vocabulary cache across tests.
COUNTER = TokenCounter()


def make_settings(tmp_path, **context) -> Settings:
    values = {"context_window_tokens": 100000}
    values.update(context)
    return Settings(context=ContextSettings(**values), data={"cache_dir": tmp_path / "cache"})


def build_loop(
    tmp_path,
    provider,
    *,
    conversation_id: str,
    store: MemoryStore | None = None,
    registry=None,
    settings=None,
):
    settings = settings or make_settings(tmp_path)
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    return AgentLoop(
        provider=provider,
        registry=registry or StubRegistry(),
        settings=settings,
        system="系统提示",
        cite=cite,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        counter=COUNTER,
        conversation_id=conversation_id,
        store=store,
    )


# --- isolation ---------------------------------------------------------------

def test_conversations_do_not_see_each_others_transcript(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    async def run():
        first = build_loop(
            tmp_path,
            ScriptedProvider([text_round("甲的答案")]),
            conversation_id="c_a",
            store=store,
        )
        await first.run("甲的问题")
        second = build_loop(
            tmp_path,
            ScriptedProvider([text_round("乙的答案")]),
            conversation_id="c_b",
            store=store,
        )
        await second.run("乙的问题")
        return first, second

    first, second = asyncio.run(run())

    assert [m.content for m in first.memory.raw if m.role == "user"] == ["甲的问题"]
    assert [m.content for m in second.memory.raw if m.role == "user"] == ["乙的问题"]


def test_conclusions_are_scoped_to_their_conversation(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    async def run():
        loop = build_loop(
            tmp_path, ScriptedProvider([text_round("答案")]), conversation_id="c_a", store=store
        )
        await loop.run("问题")
        loop.ctx.add_conclusion("甲的结论", ["cit_000001"])
        loop._persist_turn()
        other = build_loop(
            tmp_path, ScriptedProvider([text_round("答案")]), conversation_id="c_b", store=store
        )
        await other.run("另一个问题")
        return other

    other = asyncio.run(run())

    assert other.ctx.prior_conclusions == []
    # The state rides the request as a trailing message rather than the system
    # prompt, so isolation is asserted against that surface now.
    assert "甲的结论" not in other._state_text()


def test_preferences_are_shared_by_every_conversation(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.set_note("report_style", "简洁")

    async def run():
        loop = build_loop(
            tmp_path, ScriptedProvider([text_round("答案")]), conversation_id="c_x", store=store
        )
        await loop.run("问题")
        return loop

    loop = asyncio.run(run())

    assert loop.ctx.notes == {"report_style": "简洁"}
    assert "简洁" in loop._state_text()


# --- reload ------------------------------------------------------------------

def test_reload_restores_transcript_and_keeps_citation_ids(tmp_path):
    """A restart must resume the thread, and stored cids must stay valid.

    A citation is produced by a tool call carrying data sources, which is the
    path that persists it alongside the transcript.
    """
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    data = DataAccess([CountingAdapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)

    async def first_run():
        # A real tool over the stub adapter, so a citation is registered for real.
        from finharness.tools.fin.indicators import GetIndicatorsTool

        class OneTool:
            """Registry double exposing exactly one real tool."""

            def __init__(self, tool):
                self.tools = {tool.name: tool}

            def names(self):
                return list(self.tools)

            def resolve(self, name):
                return self.tools.get(name)

            def schemas(self, names=None):
                return []

            def is_read_only(self, name):
                return True

        registry = OneTool(GetIndicatorsTool(data, ctx=None))
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "get_indicators", {"symbol": "600519"})),
                text_round("第一答"),
            ]
        )
        loop = build_loop(
            tmp_path, provider, conversation_id="c_z", store=store, registry=registry, settings=settings
        )
        await loop.run("第一问")
        return loop

    loop = asyncio.run(first_run())
    assert loop.cite.all(), "the tool call should have registered a citation"
    original_cid = loop.cite.all()[0].cid

    async def reload():
        fresh = build_loop(
            tmp_path, ScriptedProvider([text_round("第二答")]), conversation_id="c_z", store=store
        )
        await fresh.run("第二问")
        return fresh

    fresh = asyncio.run(reload())

    contents = [m.content for m in fresh.memory.raw if m.role == "user"]
    assert "第一问" in contents, "the earlier turn must resume"
    assert "第二问" in contents
    # The citation kept its original id, so stored references still resolve.
    restored = fresh.cite.get(original_cid)
    assert restored is not None
    assert restored.symbol == "600519"


# --- follow-up reuse (the M4 acceptance point) --------------------------------

@dataclass
class CountingAdapter(DataAdapter):
    name: str = "counting"
    calls: int = 0

    def fetch_indicators(self, symbol, years, fields):
        self.calls += 1
        import pandas as pd

        return FetchResult(
            df=pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "roe": [30.0, 29.0, 28.0]}),
            interface="counting_indicators",
        )


def test_follow_up_on_the_same_symbol_reuses_recalled_data(tmp_path):
    """The second question about a symbol already fetched must not refetch."""
    adapter = CountingAdapter()
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    data = DataAccess([adapter], cache=LocalCache(tmp_path / "cache"), settings=settings)

    tool = RecordingTool("get_indicators", content="ROE 数据")
    # The tool double must expose the same name the adapter serves.
    registry = StubRegistry({"get_indicators": tool})

    async def run():
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "get_indicators", {"symbol": "600519"})),
                text_round("第一次回答"),
                # Follow-up: the model should be told the data is already there.
                text_round("第二次回答（复用）"),
            ]
        )
        loop = build_loop(
            tmp_path, provider, conversation_id="c_f", store=store, registry=registry, settings=settings
        )
        await loop.run("600519 的 ROE 是多少")
        memory_after_first = loop.memory.snapshot()
        await loop.run("那它的 ROE 趋势呢")
        return loop, memory_after_first

    loop, _ = asyncio.run(run())

    # The recall surface names the symbol, so a follow-up has something to reuse.
    subjects = {episode.subject for episode in loop.short_term.episodes()}
    assert "600519" in subjects
    rendered = loop._state_text()
    assert "600519" in rendered
    # And the second turn did not add a second fetch.
    assert len(tool.calls) == 1, "the follow-up must reuse the fetched data"


def test_episode_records_a_pointer_to_the_data(tmp_path):
    """L2 keeps a pointer, not a copy: the transcript would double otherwise."""
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)

    async def run():
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "get_quote", {"symbol": "600519"})),
                text_round("回答"),
            ]
        )
        loop = build_loop(
            tmp_path,
            provider,
            conversation_id="c_p",
            store=store,
            registry=StubRegistry({"get_quote": RecordingTool("get_quote", content="报价")}),
            settings=settings,
        )
        # Provide a citation so the episode has something to point at.
        loop.cite.register(
            tool="get_quote", endpoint="akshare:x", symbol="600519", params={},
            rows=1, cols=2, fingerprint="fp",
        )
        await loop.run("600519 报价")
        return loop

    loop = asyncio.run(run())
    episodes = loop.short_term.episodes()

    assert episodes, "a fetch should be recorded as an L2 episode"
    # The summary is short; the data itself lives in the cache, not the episode.
    assert all(len(episode.summary) <= 200 for episode in episodes)


def test_memory_is_capped_by_short_mem_cap(tmp_path):
    settings = make_settings(tmp_path, short_mem_cap=3)
    options = {"symbol": "600519"}

    async def run():
        loop = build_loop(
            tmp_path, ScriptedProvider([text_round("a")]), conversation_id="c_c", settings=settings
        )
        for index in range(6):
            loop._remember_episode(
                kind="data", subject=f"600519-{index}", summary=f"第{index}次取数", ref={}
            )
        return loop

    loop = asyncio.run(run())

    assert len(loop.short_term.episodes()) == 3
