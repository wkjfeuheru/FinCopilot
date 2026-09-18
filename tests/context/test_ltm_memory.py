"""跨对话长期记忆（LTM，docs 03.6.4）：情节记忆的写入、检索、注入与治理。"""

import asyncio

from finharness.config.settings import Settings
from finharness.context.memory.distill import _parse_episodes
from finharness.context.memory.store import MemoryStore
from finharness.context.session import ResearchContext
from finharness.data.citation import CitationRegistry
from finharness.tools.meta.memory import (
    ForgetMemoryTool,
    SearchMemoryTool,
    UpdateMemoryTool,
)


def make_settings(tmp_path) -> Settings:
    return Settings(data={"cache_dir": tmp_path / "cache"})


# --- 存储层：写入 / 去重 / 检索 / 保留 ---------------------------------------

def test_add_episode_is_idempotent_and_scoped_by_user(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    first = store.add_ltm_episode(
        kind="task_result", summary="茅台 2024 毛利率 91%", user_id="u1", subject="600519"
    )
    # 同内容重复写入被去重（内容哈希唯一约束）。
    again = store.add_ltm_episode(
        kind="task_result", summary="茅台 2024 毛利率 91%", user_id="u1", subject="600519"
    )
    other = store.add_ltm_episode(
        kind="task_result", summary="茅台 2024 毛利率 91%", user_id="u2", subject="600519"
    )

    assert first is not None
    assert again is None
    assert other is not None  # 去重按用户作用域，不跨用户
    assert first.ep_uid.startswith("ep_")
    assert store.list_ltm_episodes(user_id="u1")[0].summary == "茅台 2024 毛利率 91%"
    assert len(store.list_ltm_episodes(user_id="u2")) == 1
    assert store.list_ltm_episodes(user_id="u3") == []


def test_episode_summary_is_clipped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    long_text = "长" * 800

    record = store.add_ltm_episode(kind="excerpt", summary=long_text, user_id="u1")

    assert record is not None
    assert len(record.summary) <= 400
    assert record.summary.endswith("…")


def test_list_episodes_filters_by_subject_kind_and_conversation(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add_ltm_episode(
        kind="task_result", summary="茅台结论", user_id="u1",
        subject="600519", source_conversation_id="c_a",
    )
    store.add_ltm_episode(
        kind="decision", summary="决定跟踪五粮液", user_id="u1",
        subject="000858", source_conversation_id="c_b",
    )

    assert [e.subject for e in store.list_ltm_episodes(user_id="u1", subject="600519")] == ["600519"]
    assert [e.kind for e in store.list_ltm_episodes(user_id="u1", kind="decision")] == ["decision"]
    assert (
        store.list_ltm_episodes(user_id="u1", source_conversation_id="c_a")[0].summary
        == "茅台结论"
    )
    assert len(store.list_ltm_episodes(user_id="u1")) == 2


def test_get_update_delete_episode_is_user_scoped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    record = store.add_ltm_episode(kind="decision", summary="原始内容", user_id="u1")
    assert record is not None

    # 归属不符视同不存在。
    assert store.get_ltm_episode(record.ep_uid, user_id="u2") is None
    assert store.update_ltm_episode(record.ep_uid, user_id="u2", summary="x") is None
    assert store.delete_ltm_episode(record.ep_uid, user_id="u2") is False

    updated = store.update_ltm_episode(
        record.ep_uid, user_id="u1", summary="新内容", subject="600519"
    )
    assert updated is not None
    assert updated.summary == "新内容"
    assert updated.subject == "600519"
    assert store.delete_ltm_episode(record.ep_uid, user_id="u1") is True
    assert store.get_ltm_episode(record.ep_uid, user_id="u1") is None


def test_prune_ltm_episodes_by_count_keeps_newest(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    for index in range(5):
        store.add_ltm_episode(
            kind="task_result", summary=f"结论 {index}", user_id="u1"
        )

    removed = store.prune_ltm_episodes(user_id="u1", max_episodes=2, max_age_days=3650)

    assert removed == 3
    remaining = store.list_ltm_episodes(user_id="u1")
    assert {item.summary for item in remaining} == {"结论 3", "结论 4"}
    # 另一个用户的预算独立计算，不受影响。
    store.add_ltm_episode(kind="task_result", summary="别人的", user_id="u2")
    assert store.prune_ltm_episodes(user_id="u2", max_episodes=2, max_age_days=3650) == 0


def test_prune_all_ltm_episodes_covers_every_user(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    for index in range(4):
        store.add_ltm_episode(kind="task_result", summary=f"u1-{index}", user_id="u1")
    store.add_ltm_episode(kind="task_result", summary="u2-0", user_id="u2")

    removed = store.prune_all_ltm_episodes(max_episodes=1, max_age_days=3650)

    # 每个用户各保留 1 条：u1 删 3 条、u2 不删。
    assert removed == 3
    assert len(store.list_ltm_episodes(user_id="u1")) == 1
    assert len(store.list_ltm_episodes(user_id="u2")) == 1


def test_claim_user_takes_ownership_of_ltm_episodes(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add_ltm_episode(kind="decision", summary="无主情节", user_id="")

    store.claim_user("u1")

    assert store.list_ltm_episodes(user_id="u1")[0].summary == "无主情节"
    assert store.list_ltm_episodes(user_id="") == []


def test_distill_ledger_tracks_attempts_and_success(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    assert store.get_ltm_distill_state("c_1") is None
    store.mark_ltm_distilled("c_1", user_id="u1", attempts=1)
    assert store.get_ltm_distill_state("c_1") == (1, 0)
    store.mark_ltm_distilled("c_1", user_id="u1", episodes_written=3)
    assert store.get_ltm_distill_state("c_1") == (1, 3)


def test_distill_candidates_exclude_recent_and_exhausted(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    # 两个对话，都闲置（last_active_at 在很久以前）。
    for cid in ("c_old", "c_new"):
        store.ensure_conversation(cid, user_id="u1", title=cid)
    with store._connect() as connection:
        connection.execute(
            "UPDATE conversations SET last_active_at = '2000-01-01T00:00:00+00:00'"
        )

    candidates = store.ltm_distill_candidates(
        user_id="u1", max_attempts=3, idle_before="2030-01-01T00:00:00+00:00", limit=10
    )
    assert set(candidates) == {"c_old", "c_new"}

    # 已成功蒸馏的对话不再入选；失败到上限的也不再入选。
    store.mark_ltm_distilled("c_new", user_id="u1", episodes_written=2)
    store.mark_ltm_distilled("c_old", user_id="u1", attempts=3)
    assert store.ltm_distill_candidates(
        user_id="u1", max_attempts=3, idle_before="2030-01-01T00:00:00+00:00", limit=10
    ) == []


# --- 蒸馏解析 ---------------------------------------------------------------

def test_parse_episodes_accepts_json_array_and_strips_fence():
    raw = '```json\n[{"kind": "decision", "subject": "600519", "summary": "关注渠道"}]\n```'

    parsed = _parse_episodes(raw)

    assert parsed == [
        {"kind": "decision", "subject": "600519", "summary": "关注渠道", "cids": []}
    ]


def test_parse_episodes_rejects_bad_kinds_and_malformed_output():
    assert _parse_episodes("不是 JSON") == []
    # task_result 不由蒸馏产出（每轮已结构化写入），必须被过滤。
    assert _parse_episodes('[{"kind": "task_result", "summary": "x"}]') == []
    assert _parse_episodes('[{"kind": "decision", "summary": ""}]') == []
    assert _parse_episodes('[{"kind": "excerpt", "summary": "ok"}]') == [
        {"kind": "excerpt", "subject": "", "summary": "ok", "cids": []}
    ]


# --- 注入渲染 ---------------------------------------------------------------

def test_ltm_episodes_are_rendered_in_the_state_block(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add_ltm_episode(
        kind="task_result", summary="茅台 2024 毛利率 91%", user_id="u1",
        subject="600519", cids=["cit_000001"],
        source_conversation_id="c_a", source_title="茅台分析",
        source_ts="2026-01-02T00:00:00+00:00",
    )
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    ctx.store = store
    ctx.user_id = "u1"
    ctx.ltm_recent = store.list_ltm_episodes(user_id="u1")

    state = ctx.state_block()

    assert "跨对话记忆" in state
    assert "茅台 2024 毛利率 91%" in state
    assert "茅台分析" in state
    assert "cit_000001" in state
    assert "search_memory" in state


def test_state_block_has_no_ltm_section_without_episodes(tmp_path):
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)

    assert "跨对话记忆" not in ctx.state_block()


def test_refresh_ltm_recall_matches_by_symbol(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add_ltm_episode(
        kind="decision", summary="决定跟踪茅台渠道", user_id="u1", subject="600519"
    )
    store.add_ltm_episode(
        kind="decision", summary="决定跟踪五粮液", user_id="u1", subject="000858"
    )
    cite = CitationRegistry()
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=cite, settings=settings)
    ctx.store = store
    ctx.user_id = "u1"
    ctx.remember_symbol("600519")

    ctx.refresh_ltm_recall()

    assert [item.subject for item in ctx.ltm_recalled] == ["600519"]
    assert "茅台渠道" in ctx.state_block()


# --- 工具 -------------------------------------------------------------------

def test_search_memory_tool_returns_matching_episodes(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add_ltm_episode(
        kind="task_result", summary="茅台 ROE 30%", user_id="u1", subject="600519"
    )
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    ctx.store = store
    ctx.user_id = "u1"
    from finharness.data.access import DataAccess

    tool = SearchMemoryTool(DataAccess([], settings=settings), ctx=ctx)

    result = asyncio.run(tool.run(query="ROE"))
    assert result.ok is True
    assert "茅台 ROE 30%" in result.content
    # subject 过滤把不相干的标的排除。
    no_hit = asyncio.run(tool.run(subject="000858"))
    assert "没有命中任何跨对话记忆" in no_hit.content


def test_search_memory_tool_without_store_degrades(tmp_path):
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    from finharness.data.access import DataAccess

    result = asyncio.run(
        SearchMemoryTool(DataAccess([], settings=settings), ctx=ctx).run(query="x")
    )
    assert result.ok is True
    assert "未接入持久记忆" in result.content


def test_update_and_forget_tools_require_write_permission_and_work(tmp_path):
    from finharness.tools.base import PermissionLevel

    assert UpdateMemoryTool.permission is PermissionLevel.WRITE
    assert ForgetMemoryTool.permission is PermissionLevel.WRITE
    store = MemoryStore(tmp_path / "memory.db")
    record = store.add_ltm_episode(kind="decision", summary="旧", user_id="u1")
    assert record is not None
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    ctx.store = store
    ctx.user_id = "u1"
    from finharness.data.access import DataAccess

    data = DataAccess([], settings=settings)
    updated = asyncio.run(
        UpdateMemoryTool(data, ctx=ctx).run(memory_id=record.ep_uid, content="新")
    )
    assert "已更新记忆" in updated.content
    assert store.get_ltm_episode(record.ep_uid, user_id="u1").summary == "新"

    forgotten = asyncio.run(
        ForgetMemoryTool(data, ctx=ctx).run(memory_id=record.ep_uid)
    )
    assert "已删除记忆" in forgotten.content
    assert store.get_ltm_episode(record.ep_uid, user_id="u1") is None


# --- 二期：语义记忆的工具面 --------------------------------------------------

def _memory_ctx(tmp_path, store):
    from finharness.data.access import DataAccess

    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    ctx.store = store
    ctx.user_id = "u1"
    return DataAccess([], settings=settings), ctx


def test_search_memory_finds_facts_by_keyword(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(
        user_id="u1", key="maotai", statement="茅台属于白酒行业", kind="fact"
    )
    data, ctx = _memory_ctx(tmp_path, store)

    result = asyncio.run(SearchMemoryTool(data, ctx=ctx).run(query="白酒"))

    assert "茅台属于白酒行业" in result.content
    assert "[事实]" in result.content
    assert "fa_" in result.content  # 返回 id 供后续编辑/删除


def test_search_memory_can_filter_to_preferences(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.set_note("report_style", "简洁", user_id="u1")
    store.upsert_ltm_fact(user_id="u1", key="f", statement="茅台属于白酒", kind="fact")
    data, ctx = _memory_ctx(tmp_path, store)

    result = asyncio.run(SearchMemoryTool(data, ctx=ctx).run(kind="preference"))

    assert "简洁" in result.content
    assert "茅台属于白酒" not in result.content


def test_search_memory_still_returns_episodes(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add_ltm_episode(kind="decision", summary="决定跟踪渠道", user_id="u1")
    data, ctx = _memory_ctx(tmp_path, store)

    result = asyncio.run(SearchMemoryTool(data, ctx=ctx).run(kind="episode"))

    assert "决定跟踪渠道" in result.content
    assert "[关键决策]" in result.content


def test_update_memory_edits_a_fact_by_fa_uid(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    fact = store.upsert_ltm_fact(
        user_id="u1", key="report_style", statement="用表格", kind="preference"
    )
    data, ctx = _memory_ctx(tmp_path, store)

    result = asyncio.run(
        UpdateMemoryTool(data, ctx=ctx).run(memory_id=fact.fa_uid, content="简洁")
    )

    assert "已更新记忆" in result.content
    assert store.get_notes(user_id="u1") == {"report_style": "简洁"}


def test_forget_memory_removes_a_fact_by_fa_uid(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    fact = store.upsert_ltm_fact(user_id="u1", key="k", statement="v")
    data, ctx = _memory_ctx(tmp_path, store)

    result = asyncio.run(
        ForgetMemoryTool(data, ctx=ctx).run(memory_id=fact.fa_uid)
    )

    assert "已删除记忆" in result.content
    assert store.get_ltm_fact(fact.fa_uid, user_id="u1") is None


def test_unknown_memory_id_reports_not_found(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    data, ctx = _memory_ctx(tmp_path, store)

    forgot = asyncio.run(ForgetMemoryTool(data, ctx=ctx).run(memory_id="fa_nope"))
    updated = asyncio.run(
        UpdateMemoryTool(data, ctx=ctx).run(memory_id="fa_nope", content="x")
    )

    assert "未找到记忆条目" in forgot.content
    assert "未找到记忆条目" in updated.content


def test_remember_preference_indexes_the_vector_when_available(tmp_path):
    """偏好同样进入语义索引，与蒸馏产出的事实共享一套召回。"""
    from finharness.context.memory.vector import SemanticIndex, SqliteVectorStore
    from finharness.data.access import DataAccess
    from finharness.tools.meta.preference import RememberPreferenceTool

    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    ctx.store = store
    ctx.user_id = "u1"
    ctx.semantic_index = SemanticIndex(
        store=store, embedder=_StubEmbedder(), vector_store=SqliteVectorStore(store)
    )

    asyncio.run(
        RememberPreferenceTool(DataAccess([], settings=settings), ctx=ctx).run(
            key="report_style", value="简洁"
        )
    )

    assert store.get_ltm_fact_by_key(user_id="u1", key="report_style").has_embedding is True


class _StubEmbedder:
    model = "stub"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_one(self, text):
        return [1.0, 0.0]
