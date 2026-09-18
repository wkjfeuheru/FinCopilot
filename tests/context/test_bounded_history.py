"""长对话的有界加载：不把整份历史读进内存，同时不丢没人记得的历史。

这是"对话历史无限增长"在服务端的实际形态——不是每轮涨一点，而是每次重载
峰值等于全量历史。用例锁住三件事：

* 有界加载的切点落在 assistant 帧上，不产生孤儿 tool_result（否则 OpenAI
  兼容接口会拒绝该请求）；
* 被跳过的前缀必须已被摘要覆盖才允许截断，否则回退全量；
* 截断后 ``discarded`` 与真实消息序号对齐，后续压缩不会重复摘要。
"""

import asyncio

from finharness.config.settings import ContextSettings, Settings
from finharness.context.memory.store import MemoryStore
from finharness.context.session import ResearchContext
from finharness.context.tokens import TokenCounter
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.permissions.gate import PermissionGate
from finharness.types import Msg, ToolUse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import ScriptedProvider, StubRegistry, RecordingTool, text_round  # noqa: E402

COUNTER = TokenCounter()


def make_settings(tmp_path, **context) -> Settings:
    values = {"context_window_tokens": 100000}
    values.update(context)
    return Settings(context=ContextSettings(**values), data={"cache_dir": tmp_path / "cache"})


def seed_rounds(store: MemoryStore, conversation_id: str, rounds: int) -> int:
    """写入 ``rounds`` 轮 (user, assistant+工具, tool_result)，返回消息总数。"""
    messages: list[Msg] = []
    for index in range(rounds):
        messages.append(Msg.user(f"第 {index} 个问题"))
        messages.append(
            Msg(role="assistant", content=None, tool_uses=[_call(f"call_{index}")])
        )
        messages.append(
            Msg(role="tool_result", content=None, tool_results=[(f"call_{index}", "{}")])
        )
    store.ensure_conversation(conversation_id)
    store.append_messages(conversation_id, messages)
    return len(messages)


def _call(call_id: str) -> ToolUse:
    return ToolUse(call_id=call_id, name="get_quote", args={"symbol": "600519"})


def build_loop(tmp_path, provider, *, conversation_id, store, settings=None):
    settings = settings or make_settings(tmp_path)
    cite = CitationRegistry()
    ctx = ResearchContext(cite=cite, settings=settings)
    return AgentLoop(
        provider=provider,
        registry=StubRegistry(),
        settings=settings,
        system="系统提示",
        cite=cite,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        conversation_id=conversation_id,
        store=store,
        counter=COUNTER,
    )


def test_load_messages_without_limit_returns_everything(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    total = seed_rounds(store, "c_1", rounds=5)

    assert len(store.load_messages("c_1")) == total


def test_bounded_load_returns_only_the_recent_rounds(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    seed_rounds(store, "c_1", rounds=5)  # 15 条消息

    bounded = store.load_messages("c_1", recent_rounds=2)

    # 从倒数第 2 个 assistant 帧起到结尾：assistant、tool_result、user、
    # assistant、tool_result = 5 条。旧的前 10 条不再读入。
    assert len(bounded) == 5
    assert bounded[0].role == "assistant"
    assert bounded[0].tool_uses[0].call_id == "call_3"


def test_bounded_load_never_orphans_a_tool_result(tmp_path):
    """切点必须在 assistant 帧上：孤儿 tool_result 会被接口拒绝。"""
    store = MemoryStore(tmp_path / "memory.db")
    seed_rounds(store, "c_1", rounds=6)

    bounded = store.load_messages("c_1", recent_rounds=3)

    assert bounded[0].role == "assistant", bounded[0].role
    declared = {use.call_id for message in bounded for use in message.tool_uses}
    for message in bounded:
        for call_id, _ in message.tool_results:
            assert call_id in declared, f"{call_id} 的结果缺少对应的 assistant 帧"


def test_count_covered_prefix_reports_contiguous_coverage(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    seed_rounds(store, "c_1", rounds=3)
    store.add_summary_segment("c_1", seq_from=1, seq_to=6, tier=0, text="摘要")

    assert store.count_covered_prefix("c_1") == 6


def test_count_covered_prefix_stops_at_a_gap(tmp_path):
    """中间有洞就等于没覆盖到洞之后，避免跨洞误判为已覆盖。"""
    store = MemoryStore(tmp_path / "memory.db")
    seed_rounds(store, "c_1", rounds=5)
    store.add_summary_segment("c_1", seq_from=1, seq_to=3, tier=0, text="A")
    store.add_summary_segment("c_1", seq_from=8, seq_to=12, tier=0, text="B")

    assert store.count_covered_prefix("c_1") == 3


def test_history_falls_back_to_full_when_prefix_is_uncovered(tmp_path):
    """前缀没被摘要收下时不能截断——那是不声不响地遗忘。"""
    store = MemoryStore(tmp_path / "memory.db")
    total = seed_rounds(store, "c_1", rounds=5)
    settings = make_settings(tmp_path, max_loaded_rounds=2)
    loop = build_loop(tmp_path, ScriptedProvider([]), conversation_id="c_1", store=store, settings=settings)

    loaded = loop._load_recent_history()

    assert len(loaded) == total


def test_history_truncates_and_aligns_seq_when_prefix_is_covered(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    total = seed_rounds(store, "c_1", rounds=5)  # 15 条
    # 摘要覆盖前缀：前 10 条已被收下，剩下 5 条是最近两轮。
    store.add_summary_segment("c_1", seq_from=1, seq_to=10, tier=0, text="旧历史摘要")
    settings = make_settings(tmp_path, max_loaded_rounds=2)
    loop = build_loop(tmp_path, ScriptedProvider([]), conversation_id="c_1", store=store, settings=settings)

    loaded = loop._load_recent_history()

    assert len(loaded) == 5
    # 关键：discarded 必须等于真实跳过的条数，否则后续压缩的 seq_from/seq_to
    # 会从 1 重数，把已覆盖的区间再摘要一遍。
    assert loop.memory.discarded == total - len(loaded) == 10


def test_round_limit_triggers_compaction_even_below_token_budget(tmp_path):
    """轮次上限与 token 阈值并列：内容都很小、永远触不到 token 阈值时也要压缩。"""
    from finharness.context.memory.working import WorkingMemory

    settings = make_settings(
        tmp_path,
        context_window_tokens=10_000_000,  # 大到永远不会因 token 触发
        compaction_ratio=0.5,
        max_window_rounds=3,
        min_recent_rounds=1,
    )
    memory = WorkingMemory(settings=settings, counter=COUNTER)
    for index in range(5):
        memory.append_user(f"q{index}")
        memory.append_assistant(Msg(role="assistant", content=f"a{index}"))

    from finharness.context.compaction import AutoCompactor

    compactor = AutoCompactor(
        provider=ScriptedProvider([]), memory=memory, settings=settings
    )

    assert memory.over_budget(system="s", tools=[]) is False, "前提：token 未超预算"
    assert compactor.needs_compaction() is True, "轮次数超过上限也应触发压缩"
    # 且压缩确实能推进（边界不为 0），否则会每轮空转。
    removed, _ = memory.squash()
    assert removed > 0


def test_conclusions_memory_is_bounded_but_total_keeps_growing(tmp_path):
    """结论列表在内存中有上限，累计计数仍然单调（plan_id 依赖它）。"""
    from finharness.context.session import CONCLUSION_MEMORY_LIMIT

    settings = make_settings(tmp_path)
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    for index in range(CONCLUSION_MEMORY_LIMIT + 20):
        ctx.add_conclusion(f"结论 {index}", cids=[])

    assert len(ctx.conclusions) == CONCLUSION_MEMORY_LIMIT
    assert ctx._conclusion_total == CONCLUSION_MEMORY_LIMIT + 20
    # 最新的一条仍在内存中，最旧的已被裁掉。
    assert ctx.conclusions[-1].text == f"结论 {CONCLUSION_MEMORY_LIMIT + 19}"
