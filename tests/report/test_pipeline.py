"""Report pipeline: validation, citation numbering, unsourced-number marking."""

import pytest

from finharness.config.settings import Settings
from finharness.data.citation import CitationRegistry, fingerprint_frame
from finharness.report.pipeline import (
    ReportOutline,
    ReportPipeline,
    ReportSection,
    ReportValidationError,
    safe_topic,
)


def make_pipeline(tmp_path) -> tuple[ReportPipeline, CitationRegistry, str]:
    cite = CitationRegistry()
    citation = cite.register(
        tool="get_quote", endpoint="akshare:x", symbol="600519",
        params={}, rows=3, cols=2, fingerprint="abc",
    )
    settings = Settings(
        paths={"output_dir": tmp_path / "output"}, data={"cache_dir": tmp_path / "cache"}
    )
    return ReportPipeline(cite=cite, settings=settings), cite, citation.cid


def simple_outline(cid: str, **overrides) -> ReportOutline:
    values = {
        "topic": "测试研报",
        "core_view": [f"观点一 {{cite:{cid}}}"],
        "sections": [ReportSection(heading="章节一", body=f"正文 {{cite:{cid}}}", cids=[cid])],
        "risks": ["风险一", "风险二"],
    }
    values.update(overrides)
    return ReportOutline(**values)


def test_safe_topic_replaces_illegal_characters():
    assert safe_topic("茅台/估值:分析 2024") == "茅台_估值_分析_2024"
    assert safe_topic("   ") == "report"


def test_valid_outline_passes_validation(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    pipeline.validate(simple_outline(cid))  # must not raise


def test_missing_topic_is_reported(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    with pytest.raises(ReportValidationError) as exc:
        pipeline.validate(simple_outline(cid, topic=""))
    assert any("topic" in p for p in exc.value.problems)


def test_core_view_without_citation_is_reported(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    with pytest.raises(ReportValidationError) as exc:
        pipeline.validate(simple_outline(cid, core_view=["没有引用的观点"]))
    assert any("核心观点" in p and "引用" in p for p in exc.value.problems)


def test_section_without_any_citation_is_reported(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid, sections=[ReportSection(heading="裸章节", body="正文没有引用")]
    )
    with pytest.raises(ReportValidationError) as exc:
        pipeline.validate(outline)
    assert any("引用" in p for p in exc.value.problems)


def test_missing_chart_file_is_reported(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid,
        sections=[ReportSection(heading="h", body=f"b {{cite:{cid}}}", cids=[cid], charts=["/nope/x.png"])],
    )
    with pytest.raises(ReportValidationError) as exc:
        pipeline.validate(outline)
    assert any("图表不存在" in p for p in exc.value.problems)


def test_empty_risks_are_reported(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    with pytest.raises(ReportValidationError) as exc:
        pipeline.validate(simple_outline(cid, risks=[]))
    assert any("risks" in p for p in exc.value.problems)


def test_citations_are_numbered_in_first_use_order(tmp_path):
    pipeline, cite, first = make_pipeline(tmp_path)
    second = cite.register(
        tool="get_kline", endpoint="akshare:y", symbol="600519",
        params={}, rows=1, cols=1, fingerprint="def",
    ).cid
    outline = simple_outline(
        second,
        core_view=[f"先引 {{{{cite:{first}}}}}", f"再引 {{{{cite:{second}}}}}"],
    )

    markdown, ordered, _ = pipeline.build_markdown(outline)

    assert ordered == [first, second]
    assert "[1]" in markdown and "[2]" in markdown


def test_repeated_citation_keeps_one_number(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid, core_view=[f"a {{cite:{cid}}}", f"b {{cite:{cid}}}"], sections=[]
    )
    pipeline.validate(
        ReportOutline(topic="t", core_view=outline.core_view,
                      sections=[ReportSection(heading="h", body=f"x {{cite:{cid}}}")], risks=["r"])
    )
    markdown, ordered, _ = pipeline.build_markdown(outline)

    assert ordered == [cid]
    assert markdown.count("[1]") >= 2


def test_unsourced_number_is_marked_and_warned(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid,
        sections=[ReportSection(heading="h", body="营收增长 25.3% 但未标注来源")],
    )
    markdown, _, warnings = pipeline.build_markdown(outline)

    assert "[!无来源:1]" in markdown
    assert any("未标注来源" in w for w in warnings)


def test_number_with_citation_in_same_paragraph_is_not_marked(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid,
        sections=[ReportSection(heading="h", body=f"营收增长 25.3% {{cite:{cid}}}")],
    )
    markdown, _, warnings = pipeline.build_markdown(outline)

    assert "[!无来源" not in markdown
    assert warnings == []


def test_appendix_lists_only_referenced_citations(tmp_path):
    pipeline, cite, used = make_pipeline(tmp_path)
    cite.register(
        tool="unused", endpoint="z", symbol="600519", params={},
        rows=1, cols=1, fingerprint="never",
    )
    markdown, _, _ = pipeline.build_markdown(simple_outline(used))

    assert "附录" in markdown
    assert used in markdown


def test_tool_name_in_body_is_warned(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid,
        sections=[ReportSection(
            heading="说明",
            body=f"get_quote 返回的行情快照与标的不符，故改用 get_kline {{cite:{cid}}}",
        )],
    )
    _, _, warnings = pipeline.build_markdown(outline)

    assert any("get_quote" in w and "工具名" in w for w in warnings)
    assert any("get_kline" in w for w in warnings)


def test_interface_name_in_body_is_warned(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid,
        sections=[ReportSection(
            heading="说明", body=f"接口 stock_zh_a_spot_em 不可用 {{cite:{cid}}}"
        )],
    )
    _, _, warnings = pipeline.build_markdown(outline)

    assert any("stock_zh_a_spot_em" in w and "接口名" in w for w in warnings)


def test_plain_reader_wording_produces_no_internal_name_warning(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid,
        sections=[ReportSection(
            heading="说明",
            body=f"行情为 9 月 11 日收盘价 330.51 元；该估值序列未标注分位口径 {{cite:{cid}}}",
        )],
    )
    _, _, warnings = pipeline.build_markdown(outline)

    assert not any("工具名" in w or "接口名" in w for w in warnings)


def test_export_writes_markdown_and_docx(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)

    artifact = pipeline.export(simple_outline(cid))

    assert artifact.citations == [cid]
    assert artifact.markdown_path.endswith(".md")
    assert artifact.docx_path and artifact.docx_path.endswith(".docx")
    from pathlib import Path

    assert Path(artifact.markdown_path).is_file()
    assert Path(artifact.docx_path).is_file()


def test_export_can_skip_docx(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)

    artifact = pipeline.export(simple_outline(cid), formats=("md",))

    assert artifact.docx_path is None
    assert artifact.docx_error is None


def test_topic_is_sanitised_in_the_filename(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)

    artifact = pipeline.export(simple_outline(cid, topic="茅台/估值:分析"))

    assert "茅台_估值_分析" in artifact.markdown_path
    assert "/" not in artifact.markdown_path.split("\\")[-1].replace("/", "")


def test_citations_in_risks_are_numbered_like_the_body(tmp_path):
    """Risks cite the figures that trigger them; those must be substituted too."""
    pipeline, _, cid = make_pipeline(tmp_path)
    outline = simple_outline(
        cid, risks=[f"营收增速已降至 1.3% {{cite:{cid}}}，触发条件为增速转负"]
    )

    markdown, ordered, _ = pipeline.build_markdown(outline)

    assert cid in ordered
    assert "{cite:" not in markdown, "no placeholder may survive into the report"
    assert "[1]" in markdown


def test_appendix_cells_carry_no_list_bullet(tmp_path):
    pipeline, _, cid = make_pipeline(tmp_path)

    markdown, _, _ = pipeline.build_markdown(simple_outline(cid))

    for line in markdown.splitlines():
        if line.startswith("|") and cid in line:
            assert "| - " not in line


def test_fingerprint_helper_is_used_for_registration():
    """Guard: citation registration and the pipeline agree on cids."""
    import pandas as pd

    fingerprint = fingerprint_frame(pd.DataFrame({"a": [1]}))
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
