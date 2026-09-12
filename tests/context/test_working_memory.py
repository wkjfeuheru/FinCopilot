"""Working memory: transcript ownership, request size, and window maintenance."""

from finharness.config.settings import ContextSettings, Settings
from finharness.context.memory.working import KEEP_RECENT_ROUNDS, WorkingMemory
from finharness.context.tokens import TokenCounter
from finharness.types import Msg, ToolUse

# A shared counter: tiktoken's vocabulary is expensive to fetch, so tests reuse
# whatever cache the machine already has instead of requesting a fresh one.
COUNTER = TokenCounter()


def make_settings(tmp_path, **context) -> Settings:
    values = {"context_window_tokens": 1000, "compaction_ratio": 0.8}
    values.update(context)
    return Settings(context=ContextSettings(**values), data={"cache_dir": tmp_path / "cache"})


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
    # A small window makes the threshold easy to cross deterministically.
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
    # No digest is injected: earlier history lives in the summary layer.
    assert "摘要" not in [m.content for m in memory.raw if m.role == "user"]
    assert "问题4" in [m.content for m in memory.raw if m.role == "user"]
    assert "问题0" not in [m.content for m in memory.raw if m.role == "user"]
    # Two assistant frames survive: one per kept round.
    assistants = [m for m in memory.raw if m.role == "assistant"]
    assert len(assistants) == KEEP_RECENT_ROUNDS
    # The kept window starts with an assistant frame, so tool calls stay paired.
    assert memory.raw[0].role == "assistant"


def test_squash_leaves_cumulative_spend_untouched(tmp_path):
    """Compaction must not rewrite the billing figure."""
    memory = make_memory(tmp_path)
    for index in range(4):
        memory.append_user(f"问题{index}")
    before = memory.used_tokens

    memory.squash(keep_rounds=2)

    assert memory.used_tokens == before


def test_squash_reduces_the_window(tmp_path):
    memory = make_memory(tmp_path)
    # Rounds are model exchanges: an assistant frame marks each one.
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

    # A window of 1 token would otherwise be over budget immediately.
    assert memory.over_budget(system="s", tools=[]) is True


def test_rounds_are_counted_by_exchange_not_by_user_message(tmp_path):
    """A research task is one question followed by many exchanges.

    Counting user messages would find nothing foldable in exactly the case
    compaction exists for, so rounds must be counted per model exchange.
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
    """Every retained tool_result must still have its announcing assistant frame.

    A tool_result whose call_id was never announced is rejected outright by
    OpenAI-compatible APIs (HTTP 400), so the cut must land on an assistant
    frame — never between a tool call and its result.
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
