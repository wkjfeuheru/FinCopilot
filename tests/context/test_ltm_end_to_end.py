"""跨对话长期记忆的端到端行为（docs 03.6.4）：引擎写入 → 新对话召回。

这些测试走真实的 AgentLoop 与 MemoryStore（provider 为脚本替身），验证
"第一个对话形成的结论，在第二个对话里被注入"这条完整链路，以及蒸馏
双保险（扫描器 + 开新对话兜底）的选择逻辑。
"""

import asyncio
import sys
from pathlib import Path

from finharness.config.settings import Settings
from finharness.context.memory.distill import EpisodeDistiller
from finharness.context.memory.store import MemoryStore
from finharness.context.memory.vector import SemanticIndex, SqliteVectorStore
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.permissions.gate import PermissionGate
from finharness.server.distill_sweeper import distill_user_backlog, sweep_once
from finharness.tools.registry import ToolRegistry
from tests.conftest import settings_with_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import ScriptedProvider, text_round  # noqa: E402


def make_settings(tmp_path, **overrides) -> Settings:
    return settings_with_cache(tmp_path, **overrides)


def build_loop(
    tmp_path, provider, *, conversation_id, store, user_id="u1", semantic_index=None,
    auto_task_episodes=False,
):
    settings = make_settings(tmp_path, ltm={"auto_task_episodes": auto_task_episodes})
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    return AgentLoop(
        provider=provider,
        registry=ToolRegistry(DataAccess([], settings=settings), ctx=ctx, settings=settings),
        settings=settings,
        system="系统提示",
        cite=cite,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        conversation_id=conversation_id,
        store=store,
        user_id=user_id,
        semantic_index=semantic_index,
    )


def test_conclusion_persists_as_cross_conversation_episode(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    async def run():
        first = build_loop(
            tmp_path,
            ScriptedProvider([text_round("茅台结论：毛利率 91%")]),
            conversation_id="c_a",
            store=store,
            auto_task_episodes=True,
        )
        await first.run("分析茅台")
        # 模拟真实的"轮次内形成结论"：pending 非空时 _persist_turn 才落库。
        first.memory.append_user("补充一轮")
        first.ctx.add_conclusion("茅台结论：毛利率 91%", [])
        first._persist_turn()
        return first

    asyncio.run(run())

    episodes = store.list_ltm_episodes(user_id="u1", kind="task_result")
    assert len(episodes) == 1
    assert episodes[0].summary == "茅台结论：毛利率 91%"
    assert episodes[0].source_conversation_id == "c_a"
    assert episodes[0].source_title  # 对话标题随情节冗余保存


def test_task_episodes_are_not_written_by_default(tmp_path):
    """默认门控关闭：正常对话的结论不外溢成跨对话情节。

    ``auto_task_episodes`` 是唯一"无约定"的全局写入路径，默认关闭后，
    结论仍按对话隔离落库（``conclusions``），但不会成为任何新对话都能看到的记忆。
    """
    store = MemoryStore(tmp_path / "memory.db")

    async def run():
        loop = build_loop(
            tmp_path,
            ScriptedProvider([text_round("答案")]),
            conversation_id="c_a",
            store=store,
        )
        await loop.run("分析茅台")
        loop.memory.append_user("补充一轮")
        loop.ctx.add_conclusion("茅台结论：毛利率 91%", [])
        loop._persist_turn()

    asyncio.run(run())

    # 对话内结论照常保存，但没有任何跨对话情节。
    assert [c.text for c in store.load_conclusions("c_a")] == ["茅台结论：毛利率 91%"]
    assert store.list_ltm_episodes(user_id="u1", kind="task_result") == []


def test_second_conversation_receives_the_episode_in_its_request(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    provider = ScriptedProvider([text_round("第二个答案")])

    async def run():
        first = build_loop(
            tmp_path,
            ScriptedProvider([text_round("第一个答案")]),
            conversation_id="c_a",
            store=store,
            auto_task_episodes=True,
        )
        await first.run("分析茅台")
        first.memory.append_user("补充一轮")
        first.ctx.add_conclusion("茅台毛利率 91%", [])
        first._persist_turn()
        second = build_loop(tmp_path, provider, conversation_id="c_b", store=store)
        await second.run("之前茅台的结论是什么？")
        return provider

    provider = asyncio.run(run())

    # 状态作为请求末尾的临时 user 消息发送；跨对话情节应出现在其中。
    assert provider.requests
    last_request = provider.requests[-1]["messages"]
    joined = "\n".join(str(message.content or "") for message in last_request)
    assert "跨对话记忆" in joined
    assert "茅台毛利率 91%" in joined


def test_conversations_of_other_users_see_nothing(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    provider = ScriptedProvider([text_round("答案")])

    async def run():
        first = build_loop(
            tmp_path, ScriptedProvider([text_round("答案")]), conversation_id="c_a",
            store=store, user_id="u1", auto_task_episodes=True,
        )
        await first.run("分析茅台")
        first.memory.append_user("补充一轮")
        first.ctx.add_conclusion("u1 的秘密结论", [])
        first._persist_turn()
        other = build_loop(
            tmp_path, provider, conversation_id="c_b", store=store, user_id="u2",
            auto_task_episodes=True,
        )
        await other.run("我的问题")
        return provider

    provider = asyncio.run(run())

    joined = "\n".join(
        str(message.content or "") for message in provider.requests[-1]["messages"]
    )
    assert "u1 的秘密结论" not in joined


# --- 蒸馏双保险 -------------------------------------------------------------

class JsonProvider:
    """返回固定 JSON 数组的脚本 provider（模拟蒸馏产出）。"""

    model = "fake-distill"

    def __init__(self, payload: str):
        self.payload = payload
        self.requests: list[list] = []

    async def stream(self, *, system, messages, tools, usage):
        from finharness.types import ModelUsage, StreamChunk, StreamEvent

        self.requests.append(list(messages))
        yield StreamChunk(StreamEvent.TEXT_DELTA, self.payload)
        yield StreamChunk(StreamEvent.MESSAGE_END, ModelUsage(input_tokens=1, output_tokens=1))


def test_distiller_writes_decision_episodes_and_marks_ledger(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    store.ensure_conversation("c_a", user_id="u1", title="茅台分析")
    store.append_messages("c_a", [_user_msg("分析茅台渠道")])
    provider = JsonProvider(
        '[{"kind": "decision", "subject": "600519", "summary": "重点关注渠道结构"}]'
    )

    async def run():
        distiller = EpisodeDistiller(provider=provider, store=store, settings=settings)
        return await distiller.distill_conversation("c_a", user_id="u1")

    outcome = asyncio.run(run())

    assert outcome.episodes_written == 1
    assert outcome.error is None
    episodes = store.list_ltm_episodes(user_id="u1", kind="decision")
    assert episodes[0].summary == "重点关注渠道结构"
    assert episodes[0].distilled is True
    assert store.get_ltm_distill_state("c_a") == (0, 1)


def test_distiller_failure_increments_attempts_without_raising(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    store.ensure_conversation("c_a", user_id="u1")
    store.append_messages("c_a", [_user_msg("问题")])
    provider = JsonProvider("这不是 JSON")  # 解析为空 → 零情节，不报错

    async def run():
        distiller = EpisodeDistiller(provider=provider, store=store, settings=settings)
        return await distiller.distill_conversation("c_a", user_id="u1")

    outcome = asyncio.run(run())

    assert outcome.error is None  # 空产出不是错误
    assert store.get_ltm_distill_state("c_a") == (0, 0)


def test_sweep_once_distills_idle_conversations(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    store.ensure_conversation("c_idle", user_id="u1", title="闲置对话")
    store.append_messages("c_idle", [_user_msg("问题")])
    with store._connect() as connection:
        connection.execute(
            "UPDATE conversations SET last_active_at = '2000-01-01T00:00:00+00:00'"
        )
    provider = JsonProvider('[{"kind": "excerpt", "summary": "用户的思路"}]')

    class Resolver:
        def current(self, user_id):
            return provider

    async def run():
        return await sweep_once(
            store=store, provider_resolver=Resolver(), settings=settings
        )

    done = asyncio.run(run())

    assert done == 1
    assert store.list_ltm_episodes(user_id="u1", kind="excerpt")[0].summary == "用户的思路"


def test_backlog_distiller_skips_the_current_conversation(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    for cid in ("c_current", "c_old"):
        store.ensure_conversation(cid, user_id="u1", title=cid)
        store.append_messages(cid, [_user_msg("问题")])
    with store._connect() as connection:
        connection.execute(
            "UPDATE conversations SET last_active_at = '2000-01-01T00:00:00+00:00'"
        )
    provider = JsonProvider('[{"kind": "decision", "summary": "x"}]')

    async def run():
        return await distill_user_backlog(
            provider=provider, store=store, settings=settings,
            user_id="u1", exclude_conversation="c_current",
        )

    asyncio.run(run())

    assert store.get_ltm_distill_state("c_current") is None
    assert store.get_ltm_distill_state("c_old") is not None


# --- 二期：语义记忆（facts）随同一次蒸馏产出 --------------------------------

def test_distiller_writes_facts_in_the_same_call(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    store.ensure_conversation("c_a", user_id="u1", title="茅台分析")
    store.append_messages("c_a", [_user_msg("以后报告都简洁一点，少用表格")])
    provider = JsonProvider(
        '{"episodes": [{"kind": "decision", "summary": "决定用 ROE 衡量盈利"}],'
        ' "facts": [{"kind": "preference", "key": "report_style",'
        ' "statement": "报告要简洁，少用表格", "confidence": 0.9}]}'
    )

    async def run():
        distiller = EpisodeDistiller(provider=provider, store=store, settings=settings)
        return await distiller.distill_conversation("c_a", user_id="u1")

    outcome = asyncio.run(run())

    assert outcome.episodes_written == 1
    assert outcome.facts_written == 1
    # 两类记忆共用同一次 LLM 调用——这是"语义默认开启也不加成本"的前提。
    assert len(provider.requests) == 1
    assert store.get_notes(user_id="u1") == {"report_style": "报告要简洁，少用表格"}


def test_distilled_fact_confidence_is_persisted(tmp_path):
    """蒸馏产出带 confidence（0~1）时随条目落库；越界/非数字按 None 处理。"""
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    store.ensure_conversation("c_a", user_id="u1", title="茅台分析")
    store.append_messages("c_a", [_user_msg("茅台属于白酒行业")])
    provider = JsonProvider(
        '{"episodes": [], "facts": ['
        '{"kind": "fact", "key": "industry_maotai", "statement": "茅台属于白酒行业",'
        ' "confidence": 0.85},'
        '{"kind": "fact", "key": "wild_claim", "statement": "越界值丢弃",'
        ' "confidence": 1.5},'
        '{"kind": "fact", "key": "no_number", "statement": "没有数字"}]}'
    )

    async def run():
        distiller = EpisodeDistiller(provider=provider, store=store, settings=settings)
        return await distiller.distill_conversation("c_a", user_id="u1")

    outcome = asyncio.run(run())

    assert outcome.facts_written == 3
    assert store.get_ltm_fact_by_key(user_id="u1", key="industry_maotai").confidence == 0.85
    assert store.get_ltm_fact_by_key(user_id="u1", key="wild_claim").confidence is None
    assert store.get_ltm_fact_by_key(user_id="u1", key="no_number").confidence is None


def test_semantics_can_be_disabled_without_losing_episodes(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = Settings(
        ltm={"distill_semantics": False}, data={"cache_dir": tmp_path / "c"}
    )
    store.ensure_conversation("c_a", user_id="u1")
    store.append_messages("c_a", [_user_msg("问题")])
    provider = JsonProvider(
        '{"episodes": [{"kind": "decision", "summary": "a"}],'
        ' "facts": [{"kind": "preference", "key": "k", "statement": "v"}]}'
    )

    async def run():
        distiller = EpisodeDistiller(provider=provider, store=store, settings=settings)
        return await distiller.distill_conversation("c_a", user_id="u1")

    outcome = asyncio.run(run())

    assert outcome.episodes_written == 1
    assert outcome.facts_written == 0
    assert store.list_ltm_facts(user_id="u1") == []


def test_upsert_lets_a_later_conversation_override_a_preference(tmp_path):
    """冲突覆盖：第二次蒸馏给出新口径，必须赢过旧口径（用户声明优先）。"""
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    providers = [
        JsonProvider(
            '{"facts": [{"kind": "preference", "key": "report_style",'
            ' "statement": "用表格"}]}'
        ),
        JsonProvider(
            '{"facts": [{"kind": "preference", "key": "report_style",'
            ' "statement": "不用表格"}]}'
        ),
    ]

    async def run():
        for index, provider in enumerate(providers):
            cid = f"c_{index}"
            store.ensure_conversation(cid, user_id="u1")
            store.append_messages(cid, [_user_msg("问题")])
            distiller = EpisodeDistiller(
                provider=provider, store=store, settings=settings
            )
            await distiller.distill_conversation(cid, user_id="u1")

    asyncio.run(run())

    assert store.get_notes(user_id="u1") == {"report_style": "不用表格"}
    assert len(store.list_ltm_facts(user_id="u1")) == 1


def test_distilled_facts_are_indexed_when_an_embedder_is_configured(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    store.ensure_conversation("c_a", user_id="u1")
    store.append_messages("c_a", [_user_msg("问题")])
    index = SemanticIndex(
        store=store, embedder=_StubEmbedder(), vector_store=SqliteVectorStore(store)
    )
    provider = JsonProvider(
        '{"facts": [{"kind": "fact", "key": "f", "statement": "茅台属于白酒行业"}]}'
    )

    async def run():
        distiller = EpisodeDistiller(
            provider=provider, store=store, settings=settings, index=index
        )
        return await distiller.distill_conversation("c_a", user_id="u1")

    asyncio.run(run())

    assert store.get_ltm_fact_by_key(user_id="u1", key="f").has_embedding is True


# --- 二期：语义注入到请求 ---------------------------------------------------

def test_semantically_recalled_facts_are_injected(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(
        user_id="u1", key="maotai", statement="茅台属于白酒行业", kind="fact"
    )
    store.upsert_ltm_fact(user_id="u1", key="bank", statement="银行股看息差", kind="fact")
    index = SemanticIndex(
        store=store, embedder=_StubEmbedder(), vector_store=SqliteVectorStore(store)
    )
    index.index_pending(user_id="u1")
    provider = ScriptedProvider([text_round("答案")])

    async def run():
        loop = build_loop(
            tmp_path, provider, conversation_id="c_sem", store=store,
            semantic_index=index,
        )
        await loop.run("茅台属于什么行业？")
        return provider

    provider = asyncio.run(run())

    joined = "\n".join(
        str(message.content or "") for message in provider.requests[-1]["messages"]
    )
    assert "相关知识" in joined
    assert "茅台属于白酒行业" in joined
    # 语义不相关的条目不该被注入（stub 嵌入把"茅台"与其他文本分开）。
    assert "银行股看息差" not in joined


def test_no_semantic_block_without_an_embedder(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="k", statement="茅台属于白酒行业")
    provider = ScriptedProvider([text_round("答案")])

    async def run():
        loop = build_loop(tmp_path, provider, conversation_id="c_x", store=store)
        await loop.run("问题")
        return provider

    provider = asyncio.run(run())

    joined = "\n".join(
        str(message.content or "") for message in provider.requests[-1]["messages"]
    )
    # 没配 embedding 就不该出现一个空的相关知识标题。
    assert "相关知识" not in joined


def _user_msg(text: str):
    from finharness.types import Msg

    return Msg(role="user", content=text)


class _StubEmbedder:
    """确定性嵌入替身：含"茅台"的文本映射到同一方向。"""

    model = "stub"

    def embed(self, texts):
        return [[1.0, 0.0] if "茅台" in text else [0.0, 1.0] for text in texts]

    def embed_one(self, text):
        return self.embed([text])[0]


def test_distiller_marks_ledger_off_the_event_loop(tmp_path):
    """蒸馏的台账写入（mark_ltm_distilled）在工作线程执行，不占事件循环。

    空对话走最简路径：只调 mark_ltm_distilled 后返回，正好用来证明该写
    经 ``asyncio.to_thread`` 落到非主线程。
    """
    import threading

    store = MemoryStore(tmp_path / "memory.db")
    settings = make_settings(tmp_path)
    store.ensure_conversation("c_empty", user_id="u1")

    seen: dict[str, int] = {}
    original = store.mark_ltm_distilled

    def spy(*args, **kwargs):
        seen["worker"] = threading.get_ident()
        return original(*args, **kwargs)

    store.mark_ltm_distilled = spy  # type: ignore[method-assign]

    async def run():
        distiller = EpisodeDistiller(provider=JsonProvider("[]"), store=store, settings=settings)
        return await distiller.distill_conversation("c_empty", user_id="u1")

    outcome = asyncio.run(run())

    assert outcome.skipped is True
    assert seen["worker"] != threading.main_thread().ident, "台账写入必须发生在工作线程"
