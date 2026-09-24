"""长文档 map-reduce 摘要内核：分片、骨架、id 契约与乱序复原（docs 03.10）。

这些是纯逻辑测试，不涉及模型调用——因此可以精确地钉住两条设计承诺：分片不打断
结构、以及并行返回的乱序在归并前被 id 复原。
"""

from __future__ import annotations

from finharness.shared.summarize import (
    Chunk,
    MapResult,
    batch_chunks,
    chunk_id,
    document_outline,
    map_task,
    needs_layered_reduce,
    order_by_id,
    parse_map_summary,
    reduce_groups,
    reduce_task,
    split_document,
)


class FakeCounter:
    """按字符近似计数，使预算断言不依赖 tiktoken 词表。"""

    def count(self, text):
        class _C:
            tokens = max(len(text or "") // 2, 0)

        return _C()


# -- 分片 ---------------------------------------------------------------------


def test_headings_start_a_new_chunk_so_a_section_is_not_cut_in_half():
    text = "\n".join(
        [
            "非银金融行业周报",
            "一、行业观点",
            "看好券商板块的估值修复。" * 10,
            "二、公募基金净申购明细",
            "本周净申购 123 亿元。" * 10,
            "三、投资建议",
            "维持增持评级。" * 10,
        ]
    )
    # 预算小到每个小节各自成片。
    chunks = split_document(text, counter=FakeCounter(), budget_tokens=40, overlap_chars=0)

    joined_labels = " ".join(chunk.label for chunk in chunks)
    assert "一、" in joined_labels
    assert "二、" in joined_labels
    # 关键：每一个小节的标题都出现在某片里，且没有片以半个标题开头。
    for heading in ("一、行业观点", "二、公募基金净申购明细", "三、投资建议"):
        assert any(heading in chunk.text for chunk in chunks)


def test_chunk_ids_are_sequential_and_stable():
    text = "\n\n".join(f"第{i}段" + "内容" * 50 for i in range(1, 6))
    chunks = split_document(text, counter=FakeCounter(), budget_tokens=30, overlap_chars=0)

    assert [chunk.id for chunk in chunks] == [
        chunk_id(index) for index in range(1, len(chunks) + 1)
    ]
    assert chunks[0].id == "P01"


def test_adjacent_chunks_overlap_to_keep_cross_page_content():
    """重叠必须真实存在且被标注，否则跨页的表格/结论会掉进接缝。"""
    text = "\n\n".join(f"第{i}段" + "内容" * 50 for i in range(1, 5))
    chunks = split_document(text, counter=FakeCounter(), budget_tokens=30, overlap_chars=20)

    assert len(chunks) > 1
    assert chunks[0].overlap == ""  # 首片没有前文可接
    for previous, current in zip(chunks, chunks[1:]):
        assert current.overlap
        assert current.overlap == previous.text[-20:]
        assert current.text.startswith(current.overlap)


def test_a_single_oversized_section_is_hard_split_but_still_numbered():
    """单节超预算时不得整片超限，且仍逐片编号。"""
    text = "一、超长小节\n" + "内容" * 500
    chunks = split_document(text, counter=FakeCounter(), budget_tokens=40, overlap_chars=0)

    assert len(chunks) > 1
    assert [chunk.id for chunk in chunks] == [
        chunk_id(index) for index in range(1, len(chunks) + 1)
    ]


def test_empty_document_yields_no_chunks():
    assert split_document("   \n  ", counter=FakeCounter()) == []


# -- 骨架 ---------------------------------------------------------------------


def test_outline_lists_the_title_and_the_headings():
    """骨架注入每个 map 任务，是分片恢复全局语义的关键。"""
    text = "\n".join(
        ["某券商研报：2026年策略", "一、宏观", "正文", "二、行业配置", "正文", "三、风险提示", "正文"]
    )

    outline = document_outline(text)

    assert "某券商研报：2026年策略" in outline
    assert "一、宏观" in outline
    assert "三、风险提示" in outline


def test_the_map_task_carries_the_chunk_id_and_the_outline():
    chunk = Chunk(id="P03", seq=3, text="本片正文", label="三、投资建议")

    task = map_task(chunk, outline="标题：X\n  - 三、投资建议")

    # id 写进任务，并要求写进输出——这是乱序复原的契约。
    assert "P03" in task
    assert "第一行只写分片 id" in task
    assert "标题：X" in task
    assert "本片正文" in task


def test_the_map_task_flags_an_overlap_segment():
    chunk = Chunk(id="P02", seq=2, text="重叠+本片", overlap="重叠")

    task = map_task(chunk)

    assert "重叠" in task
    assert "不必重复强调" in task


# -- id 解析与乱序复原 ---------------------------------------------------------


def test_parse_reads_the_id_from_the_first_line():
    identifier, body = parse_map_summary("P04\n本片讲了公募基金净申购。", expected_id="P01")

    assert identifier == "P04"
    assert body.startswith("本片讲了")


def test_parse_tolerates_brackets_and_a_trailing_colon():
    identifier, _ = parse_map_summary("[P05]：内容", expected_id="P01")

    assert identifier == "P05"


def test_parse_falls_back_to_the_expected_id_when_the_model_omits_it():
    """漏写 id 不能让归并把该片丢掉，也不能凭空造一个顺序。"""
    identifier, body = parse_map_summary("这段摘要没有写 id。", expected_id="P07")

    assert identifier == "P07"
    assert body == "这段摘要没有写 id。"


def test_order_by_id_restores_document_order_from_shuffled_results():
    """并行返回乱序时，归并前必须按 id 复原成全文顺序。"""
    results = [
        MapResult(Chunk(id="P03", seq=3, text="c"), summary="第三片"),
        MapResult(Chunk(id="P01", seq=1, text="a"), summary="第一片"),
        MapResult(Chunk(id="P02", seq=2, text="b"), summary="第二片"),
    ]

    ordered = order_by_id(results, total=3)

    assert [item.chunk.id for item in ordered] == ["P01", "P02", "P03"]
    assert [item.summary for item in ordered] == ["第一片", "第二片", "第三片"]


def test_order_by_id_keeps_an_unrecognised_id_at_the_end():
    """无法归属的 id 补在末尾，而不是插进中间伪造一个顺序。"""
    results = [
        MapResult(Chunk(id="PX", seq=99, text="x"), summary="来历不明"),
        MapResult(Chunk(id="P01", seq=1, text="a"), summary="第一片"),
        MapResult(Chunk(id="P02", seq=2, text="b"), summary="第二片"),
    ]

    ordered = order_by_id(results, total=2)

    assert [item.chunk.id for item in ordered] == ["P01", "P02", "PX"]


# -- 分批与分层归并 -----------------------------------------------------------


def test_batches_respect_the_per_call_ceiling():
    chunks = [Chunk(id=chunk_id(i), seq=i, text="x") for i in range(1, 20)]

    batches = batch_chunks(chunks, size=8)

    assert [len(batch) for batch in batches] == [8, 8, 3]


def test_layered_reduce_triggers_when_the_combined_summaries_are_too_large():
    # FakeCounter 按 len//2 估算，故"大"输入需超过 2×预算的字符数。
    small = ["短摘要"]
    large = ["很长的摘要" * 1000]

    assert needs_layered_reduce(small, counter=FakeCounter(), budget_tokens=2000) is False
    assert needs_layered_reduce(large, counter=FakeCounter(), budget_tokens=2000) is True


def test_reduce_groups_split_the_material_for_staged_merging():
    items = [f"件{i}" for i in range(1, 10)]

    groups = reduce_groups(items, size=4)

    assert [len(group) for group in groups] == [4, 4, 1]


def test_the_reduce_task_demands_id_ordered_reconstruction():
    task = reduce_task(["[P01] 甲", "[P02] 乙"], total=2)

    assert "重建全文的逻辑顺序" in task
    # 出处必须保留，使读者可回查。
    assert "[P03]" in task
    assert "跨片综合" in task
    # 缺失的分片不得被掩盖。
    assert "缺失" in task
