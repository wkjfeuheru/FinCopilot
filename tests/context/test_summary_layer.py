"""分段摘要：累积、合并、ledger 与注入渲染。"""

from finharness.context.memory.store import MemoryStore
from finharness.context.memory.summary import SummaryLayer
from finharness.context.tokens import TokenCounter

COUNTER = TokenCounter()


def make_layer(tmp_path, *, budget: int = 0) -> tuple[SummaryLayer, MemoryStore]:
    store = MemoryStore(tmp_path / "memory.db")
    store.ensure_conversation("c_1")
    layer = SummaryLayer.load(
        conversation_id="c_1", store=store, counter=COUNTER, budget_tokens=budget
    )
    return layer, store


def test_segments_accumulate_rather_than_overwriting(tmp_path):
    """每次 compaction 追加一个 segment；更早的 segment 从不被覆盖。"""
    layer, _ = make_layer(tmp_path)

    layer.add(seq_from=1, seq_to=10, text="第一阶段：取行情")
    layer.add(seq_from=11, seq_to=20, text="第二阶段：算指标")

    assert [segment.text for segment in layer.segments] == [
        "第一阶段：取行情",
        "第二阶段：算指标",
    ]
    assert [segment.seq_from for segment in layer.segments] == [1, 11]


def test_segments_survive_a_reload(tmp_path):
    """持久化使重启后能够避免重新生成摘要。"""
    layer, store = make_layer(tmp_path)
    layer.add(seq_from=1, seq_to=10, text="第一阶段")

    reloaded = SummaryLayer.load(
        conversation_id="c_1", store=store, counter=COUNTER, budget_tokens=0
    )

    assert [segment.text for segment in reloaded.segments] == ["第一阶段"]


def test_oldest_segments_merge_while_recent_stays_faithful(tmp_path):
    """超出预算时合并*最旧*的一对；最新的 segment 保持不变。"""
    layer, _ = make_layer(tmp_path, budget=60)
    layer.add(seq_from=1, seq_to=10, text="很早以前的第一段历史" * 3)
    layer.add(seq_from=11, seq_to=20, text="稍近一些的第二段历史" * 3)
    layer.add(seq_from=21, seq_to=30, text="最近的一段")

    # 剩余两个 segment：合并后的较旧一对，加上原样保留的最近一个。
    assert len(layer.segments) == 2
    merged, recent = layer.segments
    assert recent.text == "最近的一段", "recent history must keep its fidelity"
    assert recent.tier == 0
    assert (merged.seq_from, merged.seq_to) == (1, 20)
    assert merged.tier >= 1, "older history is marked as coarser"
    # 该 layer 受其预算限制，渲染时还会进一步硬性截断。
    assert layer.tokens() <= 60
    assert COUNTER.count(layer.render(max_tokens=20)).tokens <= 20


def test_merging_preserves_the_seq_range(tmp_path):
    """合并后的 segment 覆盖两个原始 segment，因此不会丢失任何消息区间。"""
    layer, _ = make_layer(tmp_path, budget=25)
    layer.add(seq_from=5, seq_to=9, text="内容甲" * 5)
    layer.add(seq_from=10, seq_to=14, text="内容乙" * 5)
    layer.add(seq_from=15, seq_to=19, text="最新")

    merged = layer.segments[0]

    assert merged.seq_from == 5
    # 合并后的区间一直延伸到第二个原始 segment 的末尾。
    assert merged.seq_to == 14
    assert layer.segments[-1].text == "最新"


def test_ledger_aggregates_across_segments_without_duplicates(tmp_path):
    layer, _ = make_layer(tmp_path)
    layer.add(seq_from=1, seq_to=5, text="a", ledger=["get_quote(600519)"])
    layer.add(seq_from=6, seq_to=9, text="b", ledger=["get_quote(600519)", "get_kline(600519)"])

    assert layer.ledger() == ("get_quote(600519)", "get_kline(600519)")


def test_render_is_empty_without_segments(tmp_path):
    layer, _ = make_layer(tmp_path)

    assert layer.render() == ""


def test_render_includes_ranges_and_text(tmp_path):
    layer, _ = make_layer(tmp_path)
    layer.add(seq_from=1, seq_to=12, text="取了行情与指标")

    rendered = layer.render()

    assert "历史摘要" in rendered
    assert "1-12" in rendered
    assert "取了行情与指标" in rendered


def test_render_honours_a_token_budget(tmp_path):
    layer, _ = make_layer(tmp_path)
    layer.add(seq_from=1, seq_to=5, text="很长的一段历史" * 40)

    rendered = layer.render(max_tokens=20)

    assert COUNTER.count(rendered).tokens <= 21  # 预算加上标记本身


def test_layer_works_without_a_store(tmp_path):
    """未配置持久化：segment 仍会在内存中累积。"""
    layer = SummaryLayer.load(
        conversation_id="c_x", store=None, counter=COUNTER, budget_tokens=0
    )

    layer.add(seq_from=1, seq_to=3, text="内存模式")

    assert [segment.text for segment in layer.segments] == ["内存模式"]
