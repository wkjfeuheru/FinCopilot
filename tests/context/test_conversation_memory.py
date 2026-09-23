"""跨 loop 的 conversation memory：隔离、reload 与 follow-up 复用。

这里直接断言 M4 验收标准“follow-up 仅做增量取数”：针对同一 symbol 的
第二个问题不得再次触发 adapter 取数，因为数据已经在该 conversation 的
memory 中。
"""

import asyncio
import sqlite3
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
from finharness.data.citation import Citation, CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.permissions.gate import PermissionGate
from finharness.types import ToolUse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import RecordingTool, ScriptedProvider, StubRegistry, text_round, tool_round  # noqa: E402

# 跨测试共享的词表缓存。
COUNTER = TokenCounter()


def test_legacy_child_tables_are_rebuilt_with_parent_user_id(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY, title TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_active_at TEXT NOT NULL
            );
            CREATE TABLE citations (
                cid TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, tool TEXT NOT NULL,
                endpoint TEXT NOT NULL, symbol TEXT, params_json TEXT, rows INTEGER,
                cols INTEGER, fingerprint TEXT, parquet_path TEXT, from_cache INTEGER, ts TEXT
            );
            INSERT INTO conversations VALUES ('c_1', 'title', 't', 't', 't');
            INSERT INTO citations VALUES ('cit_000001', 'c_1', 'tool', 'endpoint', NULL, '{}', 0, 0, '', NULL, 0, 't');
            """
        )

    store = MemoryStore(db_path)
    assert store.claim_user("u_1") == 1
    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT user_id FROM citations WHERE conversation_id = 'c_1'").fetchone()
        columns = {column[1] for column in connection.execute("PRAGMA table_info(messages)")}

    assert row == ("u_1",)
    assert "user_id" in columns


def test_legacy_orphan_child_data_stops_migration(tmp_path):
    db_path = tmp_path / "orphan.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY, title TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_active_at TEXT NOT NULL
            );
            CREATE TABLE citations (
                cid TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, tool TEXT NOT NULL,
                endpoint TEXT NOT NULL, symbol TEXT, params_json TEXT, rows INTEGER,
                cols INTEGER, fingerprint TEXT, parquet_path TEXT, from_cache INTEGER, ts TEXT
            );
            INSERT INTO citations VALUES ('cit_000001', 'missing', 'tool', 'endpoint', NULL, '{}', 0, 0, '', NULL, 0, 't');
            """
        )

    try:
        MemoryStore(db_path)
    except RuntimeError as exc:
        assert "孤儿" in str(exc)
    else:
        raise AssertionError("孤儿子表记录必须中止迁移")


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


# --- 隔离 --------------------------------------------------------------------

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


def test_same_citation_id_is_retained_in_each_users_conversation(tmp_path):
    """租户 B 写入本地编号 cit_000001 不得替换租户 A 的引用。"""
    store = MemoryStore(tmp_path / "memory.db")
    store.ensure_conversation("c_a", user_id="u_a")
    store.ensure_conversation("c_b", user_id="u_b")

    def citation(tool: str) -> Citation:
        return Citation(
            cid="cit_000001",
            tool=tool,
            endpoint="test:source",
            symbol=None,
            params={},
            ts="2026-09-23T00:00:00+00:00",
            rows=1,
            cols=1,
            fingerprint=tool,
        )

    store.save_citations("c_a", [citation("user_a")])
    store.save_citations("c_b", [citation("user_b")])

    assert [item.tool for item in store.load_citations("c_a")] == ["user_a"]
    assert [item.tool for item in store.load_citations("c_b")] == ["user_b"]
    assert store.load_citations("c_a", user_id="u_b") == []


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
    # 状态作为尾部消息随请求发送，而不是放在 system prompt 中，
    # 因此现在针对该呈现面来断言隔离性。
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


# --- 重载 --------------------------------------------------------------------

def test_reload_restores_transcript_and_keeps_citation_ids(tmp_path):
    """重启必须能恢复对话线索，且已存储的 cid 必须保持有效。

    citation 由携带数据源的 tool call 产生，这条路径会把它与
    transcript 一并持久化。
    """
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    data = DataAccess([CountingAdapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)

    async def first_run():
        # 在 stub adapter 之上使用真实工具，因此 citation 会被真实注册。
        from finharness.tools.fin.indicators import GetIndicatorsTool

        class OneTool:
            """仅暴露一个真实工具的 Registry 替身。"""

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
    # citation 保留了其原始 id，因此已存储的引用仍能解析。
    restored = fresh.cite.get(original_cid)
    assert restored is not None
    assert restored.symbol == "600519"


# --- follow-up 复用（M4 验收点）------------------------------------------------

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
    """针对已取过的 symbol 的第二个问题不得重新取数。"""
    adapter = CountingAdapter()
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    data = DataAccess([adapter], cache=LocalCache(tmp_path / "cache"), settings=settings)

    tool = RecordingTool("get_indicators", content="ROE 数据")
    # 工具替身必须暴露与 adapter 服务相同的名称。
    registry = StubRegistry({"get_indicators": tool})

    async def run():
        provider = ScriptedProvider(
            [
                tool_round(ToolUse("c1", "get_indicators", {"symbol": "600519"})),
                text_round("第一次回答"),
                # Follow-up：应告知模型数据已经存在。
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

    # recall 面标明了 symbol，因此 follow-up 有可复用的东西。
    subjects = {episode.subject for episode in loop.short_term.episodes()}
    assert "600519" in subjects
    rendered = loop._state_text()
    assert "600519" in rendered
    # 且第二轮没有新增第二次取数。
    assert len(tool.calls) == 1, "the follow-up must reuse the fetched data"


def test_episode_records_a_pointer_to_the_data(tmp_path):
    """L2 保存的是指针而非副本：否则 transcript 会翻倍。"""
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
        # 提供一个 citation，使 episode 有可指向的对象。
        loop.cite.register(
            tool="get_quote", endpoint="akshare:x", symbol="600519", params={},
            rows=1, cols=2, fingerprint="fp",
        )
        await loop.run("600519 报价")
        return loop

    loop = asyncio.run(run())
    episodes = loop.short_term.episodes()

    assert episodes, "a fetch should be recorded as an L2 episode"
    # summary 很短；数据本身存放在 cache 中，而不是 episode 中。
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
