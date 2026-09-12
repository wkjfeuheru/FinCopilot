"""Segmented summaries: accumulation, merging, ledger, injection rendering."""

from finharness.config.settings import Settings
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
    """Each compaction appends a segment; earlier ones are never overwritten."""
    layer, _ = make_layer(tmp_path)

    layer.add(seq_from=1, seq_to=10, text="第一阶段：取行情")
    layer.add(seq_from=11, seq_to=20, text="第二阶段：算指标")

    assert [segment.text for segment in layer.segments] == [
        "第一阶段：取行情",
        "第二阶段：算指标",
    ]
    assert [segment.seq_from for segment in layer.segments] == [1, 11]


def test_segments_survive_a_reload(tmp_path):
    """Persistence is what makes a restart avoid re-summarising."""
    layer, store = make_layer(tmp_path)
    layer.add(seq_from=1, seq_to=10, text="第一阶段")

    reloaded = SummaryLayer.load(
        conversation_id="c_1", store=store, counter=COUNTER, budget_tokens=0
    )

    assert [segment.text for segment in reloaded.segments] == ["第一阶段"]


def test_oldest_segments_merge_while_recent_stays_faithful(tmp_path):
    """Over budget the *oldest* pair merges; the newest segment is untouched."""
    layer, _ = make_layer(tmp_path, budget=60)
    layer.add(seq_from=1, seq_to=10, text="很早以前的第一段历史" * 3)
    layer.add(seq_from=11, seq_to=20, text="稍近一些的第二段历史" * 3)
    layer.add(seq_from=21, seq_to=30, text="最近的一段")

    # Two segments remain: the merged older pair, plus the recent one verbatim.
    assert len(layer.segments) == 2
    merged, recent = layer.segments
    assert recent.text == "最近的一段", "recent history must keep its fidelity"
    assert recent.tier == 0
    assert (merged.seq_from, merged.seq_to) == (1, 20)
    assert merged.tier >= 1, "older history is marked as coarser"
    # The layer is bounded by its budget, and rendering caps it hard.
    assert layer.tokens() <= 60
    assert COUNTER.count(layer.render(max_tokens=20)).tokens <= 20


def test_merging_preserves_the_seq_range(tmp_path):
    """A merged segment spans both originals, so no message range is lost."""
    layer, _ = make_layer(tmp_path, budget=25)
    layer.add(seq_from=5, seq_to=9, text="内容甲" * 5)
    layer.add(seq_from=10, seq_to=14, text="内容乙" * 5)
    layer.add(seq_from=15, seq_to=19, text="最新")

    merged = layer.segments[0]

    assert merged.seq_from == 5
    # The merged span reaches the end of the second original segment.
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

    assert COUNTER.count(rendered).tokens <= 21  # budget plus the marker


def test_layer_works_without_a_store(tmp_path):
    """No persistence configured: segments still accumulate in memory."""
    layer = SummaryLayer.load(
        conversation_id="c_x", store=None, counter=COUNTER, budget_tokens=0
    )

    layer.add(seq_from=1, seq_to=3, text="内存模式")

    assert [segment.text for segment in layer.segments] == ["内存模式"]
