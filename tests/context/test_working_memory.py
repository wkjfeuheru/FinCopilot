"""Working memory：transcript 归属、请求大小与窗口维护。"""

from finharness.config.settings import ContextSettings, Settings
from finharness.context.memory.working import KEEP_RECENT_ROUNDS, WorkingMemory
from finharness.context.tokens import TokenCounter
from finharness.types import Msg, ToolUse
from tests.conftest import settings_with_cache

# 共享的 counter：tiktoken 的词表获取代价高昂，因此测试复用机器上
# 已有的缓存，而不是每次重新请求一份新的。
COUNTER = TokenCounter()


def make_settings(tmp_path, **context) -> Settings:
    values = {"context_window_tokens": 1000, "compaction_ratio": 0.8}
    values.update(context)
    return settings_with_cache(tmp_path, context=ContextSettings(**values))


def make_memory(tmp_path, **context) -> WorkingMemory:
    return WorkingMemory(settings=make_settings(tmp_path, **context), counter=COUNTER)


def test_append_tracks_cumulative_tokens(tmp_path):
    memory = make_memory(tmp_path)
    memory.append_user("你好")
    first = memory.used_tokens
    memory.append_assistant(Msg(role="assistant", content="世界"))

    assert first > 0
    assert memory.used_tokens > first


def test_snapshot_is_a_copy(tmp_path):
    memory = make_memory(tmp_path)
    memory.append_user("问题")

    snapshot = memory.snapshot()
    snapshot.append(Msg.user("外来的"))

    assert len(memory.raw) == 1


def test_request_tokens_counts_system_messages_and_schemas(tmp_path):
    memory = make_memory(tmp_path)
    memory.append_user("贵州茅台最新股价是多少")
    tools = [{"type": "function", "function": {"name": "get_quote", "description": "查询报价", "parameters": "{}"}}]

    with_tools = memory.request_tokens(system="系统提示", tools=tools)
    without_tools = memory.request_tokens(system="系统提示", tools=[])

    assert with_tools > without_tools


def test_not_over_budget_below_the_threshold(tmp_path):
    memory = make_memory(tmp_path, context_window_tokens=100000)
    memory.append_user("短问题")

    assert memory.over_budget(system="s", tools=[]) is False


def test_over_budget_past_the_ratio(tmp_path):
    # 较小的窗口能让阈值被确定性地轻松越过。
    memory = make_memory(tmp_path, context_window_tokens=200, compaction_ratio=0.5)
    for _ in range(40):
        memory.append_user("这是一段足够长的中文文本用来把窗口撑满" * 3)

    assert memory.over_budget(system="s", tools=[]) is True


def test_squash_keeps_the_recent_rounds_and_returns_removed_count(tmp_path):
    memory = make_memory(tmp_path)
    for index in range(5):
        memory.append_user(f"问题{index}")
        memory.append_assistant(Msg(role="assistant", content=f"回答{index}"))

    removed, discarded = memory.squash(keep_rounds=2)

    assert removed > 0
    # 不注入 digest：更早的历史存放在 summary layer 中。
    assert "摘要" not in [m.content for m in memory.raw if m.role == "user"]
    assert "问题4" in [m.content for m in memory.raw if m.role == "user"]
    assert "问题0" not in [m.content for m in memory.raw if m.role == "user"]
    # 保留两个 assistant frame：每个保留的 round 一个。
    assistants = [m for m in memory.raw if m.role == "assistant"]
    assert len(assistants) == KEEP_RECENT_ROUNDS
    # 保留的窗口以 assistant frame 开头，因此 tool call 与其结果保持配对。
    assert memory.raw[0].role == "assistant"


def test_squash_leaves_cumulative_spend_untouched(tmp_path):
    """Compaction 不得改写计费数值。"""
    memory = make_memory(tmp_path)
    for index in range(4):
        memory.append_user(f"问题{index}")
    before = memory.used_tokens

    memory.squash(keep_rounds=2)

    assert memory.used_tokens == before


def test_squash_reduces_the_window(tmp_path):
    memory = make_memory(tmp_path)
    # Round 指模型的一轮交互：每个 assistant frame 标记一个 round。
    for index in range(6):
        memory.append_user("很长的历史内容" * 20)
        memory.append_assistant(Msg(role="assistant", content="很长的阶段回答" * 20))
    before = memory.request_tokens(system="s", tools=[])

    memory.squash(keep_rounds=2)

    assert memory.request_tokens(system="s", tools=[]) < before


def test_squash_with_nothing_to_fold_is_a_noop(tmp_path):
    memory = make_memory(tmp_path)
    memory.append_user("只有一个回合")
    memory.append_assistant(Msg(role="assistant", content="唯一回答"))
    before = len(memory.raw)

    removed, _ = memory.squash(keep_rounds=2)

    assert removed == 0
    assert len(memory.raw) == before


def test_zero_window_disables_the_check(tmp_path):
    memory = make_memory(tmp_path, context_window_tokens=1)
    memory.append_user("内容")

    # 否则 1 token 的窗口会立即处于超预算状态。
    assert memory.over_budget(system="s", tools=[]) is True


def test_rounds_are_counted_by_exchange_not_by_user_message(tmp_path):
    """一次研究任务是一个问题后跟多轮交互。

    若按 user message 计数，恰好在 compaction 存在的那个场景下会找不到任何
    可折叠的内容，因此必须按模型交互轮次来计数 round。
    """
    memory = make_memory(tmp_path)
    memory.append_user("只问了一次很长的问题" * 10)
    for index in range(5):
        memory.append_assistant(Msg(role="assistant", content=f"第{index}轮" * 10))
        memory.append(
            Msg(role="tool_result", content=None, tool_results=[(f"c{index}", "数据" * 10)])
        )

    removed, _ = memory.squash(keep_rounds=2)

    assert removed > 0, "a single-question long session must still be foldable"
    assert memory.raw[0].role == "assistant"


def test_squash_never_orphans_a_tool_result(tmp_path):
    """每个保留的 tool_result 都必须仍能找到声明它的 assistant frame。

    call_id 从未被声明过的 tool_result 会被 OpenAI 兼容 API 直接拒绝
    （HTTP 400），因此切割点必须落在 assistant frame 上——绝不能落在
    tool call 与其结果之间。
    """
    memory = make_memory(tmp_path)
    memory.append_user("问题")
    for index in range(6):
        memory.append_assistant(
            Msg(
                role="assistant",
                content=None,
                tool_uses=[ToolUse(call_id=f"c{index}", name="get_quote", args={})],
            )
        )
        memory.append(
            Msg(role="tool_result", content=None, tool_results=[(f"c{index}", "结果")])
        )

    memory.squash(keep_rounds=2)

    announced: set[str] = set()
    for message in memory.raw:
        for tool_use in message.tool_uses:
            announced.add(tool_use.call_id)
        for call_id, _ in message.tool_results:
            assert call_id in announced, f"tool_result {call_id} has no assistant frame"
    assert memory.raw[0].role == "assistant", "the kept window must start with a tool call"
