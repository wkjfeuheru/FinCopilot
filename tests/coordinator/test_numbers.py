"""数字核对内核：缺值识别、按页定位与补齐任务构造（docs 03.10）。

这些是纯逻辑测试，不涉及模型调用，因此可以精确钉住两条设计承诺：只把**确证的缺值**
判为缺口（正常行文与已给值的量词都不算），以及定位不到时宁可返回空而不猜页码。
"""

from __future__ import annotations

from finharness.shared.numbers import (
    CLIP_MARKER,
    MAX_GAPS,
    find_gaps,
    page_span,
    pages_for_anchor,
    repair_task,
)

# -- 缺口识别 -----------------------------------------------------------------


def test_a_clip_marker_is_a_gap():
    """分片摘要的裁剪标记是本工具自己留下的确证缺口。"""
    text = f"社融：8 月新增 1.66 万亿元，同比少增{CLIP_MARKER}"

    gaps = find_gaps(text)

    assert len(gaps) == 1
    # 锚落在标记之前的原话上，且不含省略号。
    assert "同比少增" in gaps[0].anchor
    assert "…" not in gaps[0].anchor


def test_a_cue_followed_by_ellipsis_is_a_gap():
    """量词后紧跟省略号：模型明确表示"此处应有数值"。"""
    gaps = find_gaps("社融：8 月新增 1.66 万亿元，同比少增…")

    assert len(gaps) == 1
    assert "同比少增" in gaps[0].cue


def test_a_number_cut_off_mid_digits_is_a_gap():
    """数值断在中间（"分别扩大 1.4…"）也是缺口——真实研报里最常见的一种。"""
    gaps = find_gaps("8 月单月销售面积同比-14.8%，降幅较 7 月分别扩大 1.4…")

    assert len(gaps) == 1
    # 锚含半个数字，正是它把定位带回原文的"扩大 1.4pct"。
    assert gaps[0].anchor.endswith("1.4")


def test_a_cue_with_a_real_value_is_not_a_gap():
    """已给出具体数值的量词不是缺口——否则正常摘要会被成批误报。"""
    text = "8 月新增 1.66 万亿元，同比少增 8600 亿元，降幅扩大 3.1pct。"

    assert find_gaps(text) == []


def test_plain_prose_without_quantities_is_not_a_gap():
    assert find_gaps("本报告认为行业格局正在改善，龙头优势稳固。") == []


def test_a_stylistic_pause_ellipsis_is_not_a_gap():
    """行文里的省略号（前无数字、后无量词）不是缺口，收进来只会制造噪声。"""
    assert find_gaps("篇幅所限，其余章节从略……详见原文。") == []


def test_a_dangling_cue_before_a_period_is_not_a_gap():
    """悬空量词后跟句号是正常行文（如标题），不判为缺口。"""
    assert find_gaps("二、拿地规模持续收缩，开工竣工降幅均扩大") == []


def test_gaps_are_deduplicated_by_anchor():
    """同一句话重复出现（分片重叠）只报一次。"""
    line = f"同比少增{CLIP_MARKER}"
    gaps = find_gaps("\n".join([line, line]))

    assert len(gaps) == 1


def test_the_number_of_gaps_is_capped():
    """异常输入不得放大下游派发——缺口数必须有上限。"""
    text = "\n".join(f"第{index}项同比少增{CLIP_MARKER}" for index in range(40))

    assert len(find_gaps(text)) == MAX_GAPS


def test_an_empty_text_yields_no_gaps():
    assert find_gaps("") == []


# -- 按页定位 -----------------------------------------------------------------


def test_an_anchor_is_located_on_its_page():
    pages = ["封面与免责声明", "8 月社融新增 1.66 万亿元，同比少增 8600 亿元。"]

    assert pages_for_anchor(pages, "8月社融新增1.66万亿元，同比少增") == [2]


def test_location_survives_whitespace_differences():
    """PDF 抽取会在数字与单位之间插空格，定位必须先归一空白。"""
    pages = ["无关", "8 月 单月 销售 面积 同比 -14.8%， 降幅 较 7 月 扩大 1.4pct"]

    assert pages_for_anchor(pages, "8月单月销售面积同比-14.8%，降幅较7月扩大") == [2]


def test_location_falls_back_by_shortening_the_anchor_tail():
    """锚的前缀可能跨了页眉页脚，逐级缩短锚尾应仍能命中。"""
    pages = ["页眉噪声\n同比少增 8600 亿元"]

    assert pages_for_anchor(pages, "完全不同的前缀内容同比少增") == [1]


def test_an_unlocatable_anchor_returns_nothing_rather_than_a_guess():
    pages = ["第 1 页", "第 2 页"]

    assert pages_for_anchor(pages, "这段内容在任何页面都不存在") == []


def test_a_too_short_anchor_is_refused():
    """短到没有定位价值的锚宁可放弃，也不乱指页码。"""
    assert pages_for_anchor(["同比少增 1 亿元"], "同比") == []


def test_hits_are_limited_and_ordered():
    pages = ["同比少增 A", "同比少增 B", "同比少增 C", "同比少增 D"]

    assert pages_for_anchor(pages, "同比少增", limit=2) == [1, 2]


# -- 页码区间与补齐任务 -------------------------------------------------------


def test_page_span_collapses_to_one_readable_range():
    """read_pdf 只接受单个连续区间，多页命中必须收敛成一段。"""
    assert page_span([3]) == "3"
    assert page_span([2, 3]) == "2-3"
    # 跨度超过单次上限时从首片起读并截到 10 页。
    assert page_span([1, 40]) == "1-10"


def test_the_repair_task_is_self_contained():
    """spawn 的契约：worker 拿不到主上下文，任务必须自带材料位置与指令。"""
    task = repair_task(path="D:/data/cache/pdf/r.pdf", pages=[4, 5], cue="同比少增")

    assert "D:/data/cache/pdf/r.pdf" in task
    assert "read_pdf" in task
    assert "4-5" in task
    assert "同比少增" in task
    # 明确禁止编造与换算，这是数字核对的底线。
    assert "不要推断" in task or "不得猜测" in task
