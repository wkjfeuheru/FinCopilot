"""Review 编排：latch、降级以及 audit metadata（文档 03.10）。

触发点——位于 write_report 内部——在 test_artifacts.py 中针对真实 tool 做了端到端
测试；这些测试则直接检验编排模块：digest latch、每条降级路径，
以及 audit hook 读取的 metadata。
"""

import asyncio
from pathlib import Path

from finharness.coordinator.review import (
    REVIEW_DONE,
    REVIEW_REUSED,
    REVIEW_SKIPPED,
    REVIEW_UNREVIEWED,
    ReviewOutcome,
    declared_clear,
    format_review_lines,
    parse_findings,
    review_report,
)


class FakeCoordinator:
    """替代 sub-agent：记录请求，返回预设意见。"""

    def __init__(self, *, summary="### [中] 风险章节缺触发条件", ok=True,
                 error=None, raises=None):
        self.summary = summary
        self.ok = ok
        self.error = error
        self.raises = raises
        self.calls: list[dict] = []

    async def review_risk(self, *, topic, markdown):
        self.calls.append({"topic": topic, "markdown": markdown})
        if self.raises is not None:
            raise self.raises
        from finharness.coordinator import SubAgentResult

        return SubAgentResult(
            focus="risk", summary=self.summary, ok=self.ok, error=self.error,
            input_tokens=10, output_tokens=4, turns=1,
        )


def write_report(tmp_path, body="# 报告\n\n正文。\n\n## 附录\n\n引用表。") -> Path:
    markdown = tmp_path / "报告_20260914.md"
    markdown.write_text(body, encoding="utf-8")
    return markdown


def run_review(coordinator, *, topic="测试报告", markdown_path=None):
    return asyncio.run(
        review_report(coordinator, topic=topic, markdown_path=markdown_path)
    )


def test_a_successful_review_records_comments_and_the_sidecar(tmp_path):
    coordinator = FakeCoordinator()
    markdown = write_report(tmp_path)

    outcome = run_review(coordinator, markdown_path=markdown)

    assert outcome.status == REVIEW_DONE
    assert outcome.comments == FakeCoordinator().summary
    review = Path(outcome.review_path)
    assert review.is_file()
    text = review.read_text(encoding="utf-8")
    assert text.startswith("<!-- reviewed-body:")
    assert "风险终审意见" in text
    assert "风险章节缺触发条件" in text
    # reviewer 评判的是正文，而非引用附录。
    assert "附录" not in coordinator.calls[0]["markdown"]


def test_the_digest_latch_reuses_a_current_review_instead_of_paying_twice(tmp_path):
    first = FakeCoordinator()
    markdown = write_report(tmp_path)
    run_review(first, markdown_path=markdown)

    second = FakeCoordinator()
    outcome = run_review(second, markdown_path=markdown)

    assert outcome.status == REVIEW_REUSED
    assert second.calls == []
    assert outcome.review_path == str(markdown.with_suffix(".review.md"))
    lines = format_review_lines(outcome)
    assert "复用" in lines[0] or any("复用" in line for line in lines)
    # 复用不等于通过：既有意见里的高严重度条目要照旧被读出并阻断。
    assert outcome.high_severity_titles() == ("风险章节缺触发条件",) or outcome.findings
    # 结果里不再出现 sidecar 路径（docs 03.10.7：终审对用户不可见）。
    assert str(markdown.with_suffix(".review.md")) not in "\n".join(lines)


def test_a_revised_body_earns_a_fresh_review(tmp_path):
    first = FakeCoordinator()
    markdown = write_report(tmp_path)
    run_review(first, markdown_path=markdown)

    markdown.write_text("# 报告\n\n修订后的正文。\n\n## 附录\n\n引用表。", encoding="utf-8")
    second = FakeCoordinator()
    outcome = run_review(second, markdown_path=markdown)

    assert outcome.status == REVIEW_DONE
    assert len(second.calls) == 1


def test_a_raising_reviewer_degrades_to_unreviewed(tmp_path):
    coordinator = FakeCoordinator(raises=RuntimeError("provider down"))
    markdown = write_report(tmp_path)

    outcome = run_review(coordinator, markdown_path=markdown)

    assert outcome.status == REVIEW_UNREVIEWED
    assert "RuntimeError" in (outcome.error or "")
    lines = format_review_lines(outcome)
    assert "风险终审未完成" in lines[0]
    assert "未经独立复核" in lines[0]


def test_a_declined_review_degrades_to_unreviewed(tmp_path):
    coordinator = FakeCoordinator(ok=False, error="max_turns_exhausted")
    markdown = write_report(tmp_path)

    outcome = run_review(coordinator, markdown_path=markdown)

    assert outcome.status == REVIEW_UNREVIEWED
    assert outcome.error == "max_turns_exhausted"
    assert outcome.error in format_review_lines(outcome)[0]


def test_an_unreadable_body_degrades_to_unreviewed(tmp_path):
    """读取失败不应连累报告——不做 review，但也不抛异常。"""
    coordinator = FakeCoordinator()

    outcome = run_review(coordinator, markdown_path=tmp_path / "missing.md")

    assert outcome.status == REVIEW_UNREVIEWED
    assert outcome.error
    assert coordinator.calls == []


def test_no_coordinator_is_a_skip_that_stays_silent_in_the_result(tmp_path):
    markdown = write_report(tmp_path)

    outcome = run_review(None, markdown_path=markdown)

    assert outcome.status == REVIEW_SKIPPED
    assert format_review_lines(outcome) == []


def test_a_failed_sidecar_write_still_delivers_the_comments(tmp_path):
    """丢失 sidecar 文件不应丢失 review 本身。"""
    coordinator = FakeCoordinator()
    markdown = write_report(tmp_path)
    # 在 sidecar 文件应在的位置放一个目录，使写入失败。
    markdown.with_suffix(".review.md").mkdir()

    outcome = run_review(coordinator, markdown_path=markdown)

    assert outcome.status == REVIEW_DONE
    assert outcome.comments == FakeCoordinator().summary
    assert outcome.review_path is None
    assert outcome.error and outcome.error.startswith("OSError")
    lines = format_review_lines(outcome)
    assert "风险终审意见" in lines[0]
    assert "终审意见落盘失败" in "\n".join(lines)


def test_empty_review_comments_render_as_no_substantive_findings(tmp_path):
    coordinator = FakeCoordinator(summary="")
    markdown = write_report(tmp_path)

    outcome = run_review(coordinator, markdown_path=markdown)

    assert outcome.status == REVIEW_DONE
    assert "未发现实质性问题" in "\n".join(format_review_lines(outcome))


def test_a_reviewer_supplied_title_is_not_duplicated_in_the_sidecar(tmp_path):
    """模板会写出 H1；若 reviewer 重复该标题，不得使其重复出现。"""
    coordinator = FakeCoordinator(
        summary="# 风险终审意见：测试报告\n\n### [高] 正文数字缺引用"
    )
    markdown = write_report(tmp_path)

    outcome = run_review(coordinator, markdown_path=markdown)

    text = Path(outcome.review_path).read_text(encoding="utf-8")
    assert text.count("风险终审意见") == 1
    assert "### [高] 正文数字缺引用" in text


def test_a_section_heading_from_the_reviewer_is_kept(tmp_path):
    """只丢弃多余的 H1；reviewer 自己的章节保留。"""
    coordinator = FakeCoordinator(summary="## 一、独立核对结果\n\n全部一致。")
    markdown = write_report(tmp_path)

    outcome = run_review(coordinator, markdown_path=markdown)

    text = Path(outcome.review_path).read_text(encoding="utf-8")
    assert "## 一、独立核对结果" in text
    assert "全部一致。" in text


def test_outcome_metadata_carries_the_audit_payload(tmp_path):
    """audit hook 的 review 行由该 dict 构建，因此其结构是一份契约。"""
    coordinator = FakeCoordinator(ok=False, error="max_turns_exhausted")
    markdown = write_report(tmp_path)

    outcome = run_review(coordinator, markdown_path=markdown)

    meta = outcome.audit_metadata()
    assert meta["status"] == REVIEW_UNREVIEWED
    assert meta["topic"] == "测试报告"
    assert meta["error"] == "max_turns_exhausted"
    assert meta["review_path"] is None
    # 引擎侧的两个键供会话状态使用；审计行只取前四个键，因此这是增量而非破坏。
    assert "终审未完成" in meta["unresolved_note"]
    assert meta["high_severity_titles"] == []


# -- 严重度解析与未结事项（docs 03.10.7）------------------------------------


def outcome_with(comments: str, *, status: str = REVIEW_DONE) -> ReviewOutcome:
    return ReviewOutcome(status=status, topic="测试报告", comments=comments)


def test_conventional_findings_are_parsed_by_severity():
    comments = (
        "### [高] 营收数字缺引用\n- 位置：财务摘要\n\n"
        "### [中] 风险章缺触发条件\n- 位置：风险提示\n\n"
        "### [低] 未标注报告期\n"
    )

    findings = parse_findings(comments)

    assert [(f.severity, f.title) for f in findings] == [
        ("高", "营收数字缺引用"),
        ("中", "风险章缺触发条件"),
        ("低", "未标注报告期"),
    ]


def test_only_high_severity_findings_block_delivery():
    """中/低意见照旧附给模型，但不构成未结事项——否则闸门会被文风级意见淹没。"""
    outcome = outcome_with("### [中] 风险章缺触发条件\n### [低] 未标注报告期")

    assert outcome.high_severity_titles() == ()
    assert outcome.unresolved_note() is None
    assert "不得视为已完成" not in "\n".join(format_review_lines(outcome))


def test_a_high_severity_finding_blocks_with_its_title_listed():
    outcome = outcome_with("### [高] 营收数字缺引用\n- 位置：财务摘要")

    lines = format_review_lines(outcome)

    assert "不得视为已完成" in lines[0]
    assert "1 项" in lines[0]
    assert "[高] 营收数字缺引用" in "\n".join(lines)
    assert outcome.unresolved_note() == "测试报告：[高] 营收数字缺引用"


def test_the_clear_line_is_matched_exactly_not_as_prose():
    """转述“未发现实质性问题”不能算通过，否则一句话就能伪造一份干净的终审。"""
    outcome = outcome_with("复核者认为本文未发现实质性问题，但结论缺少引用。")

    assert outcome.findings == ()
    assert outcome.unclassified() is True
    assert outcome.unresolved_note() is not None
    assert "未能解析严重度" in "\n".join(format_review_lines(outcome))


def test_a_markdown_decorated_clear_line_still_counts_as_clear():
    """写成 **未发现实质性问题** 时若判为无法解析，会阻断一份没有问题的报告，
    而正文未变又命中内容闩锁复用同一条意见——那处阻断走不出去。"""
    for comments in ("**未发现实质性问题**", "## 未发现实质性问题", "未发现实质性问题。"):
        outcome = outcome_with(comments)

        assert declared_clear(comments) is True, comments
        assert outcome.unclassified() is False, comments
        assert outcome.unresolved_note() is None, comments


def test_an_explicit_clear_verdict_is_not_an_unresolved_item():
    outcome = outcome_with("未发现实质性问题")

    assert outcome.unclassified() is False
    assert outcome.unresolved_note() is None
    assert "不得视为已完成" not in "\n".join(format_review_lines(outcome))


def test_empty_comments_do_not_block():
    """空意见沿用既有语义（未发现问题），不按“未分类”处理。"""
    outcome = outcome_with("")

    assert outcome.unclassified() is False
    assert outcome.unresolved_note() is None
    assert "未发现实质性问题" in "\n".join(format_review_lines(outcome))


def test_unparseable_comments_are_treated_as_unresolved():
    outcome = outcome_with("1. 营收数字缺引用（严重度：高）\n2. 风险章不完整")

    lines = format_review_lines(outcome)

    assert outcome.unclassified() is True
    assert "未能解析严重度" in lines[0]
    assert outcome.unresolved_note() is not None


def test_a_reused_review_re_reads_its_findings_from_the_sidecar(tmp_path):
    """复用路径必须重新读出严重度：正文没改，高严重度问题就还在（docs 03.10.7）。"""
    coordinator = FakeCoordinator(summary="### [高] 营收数字缺引用")
    markdown = write_report(tmp_path)
    run_review(coordinator, markdown_path=markdown)

    reused = run_review(FakeCoordinator(), markdown_path=markdown)

    assert reused.status == REVIEW_REUSED
    assert reused.high_severity_titles() == ("营收数字缺引用",)
    assert "不得视为已完成" in format_review_lines(reused)[0]


def test_the_write_report_tool_declares_the_review_outcome_in_metadata(tmp_path):
    """经由 tool 端到端验证：audit hook 读取的 metadata 存在。"""
    from finharness.config.settings import Settings
    from finharness.context.session import ResearchContext
    from finharness.data.access import DataAccess
    from finharness.data.citation import CitationRegistry
    from finharness.tools.fin.writer import WriteReportTool

    settings = Settings(
        paths={"output_dir": tmp_path / "output"}, data={"cache_dir": tmp_path / "cache"}
    )
    cite = CitationRegistry()
    cid = cite.register(
        tool="get_quote", endpoint="e", symbol="600519", params={},
        rows=1, cols=1, fingerprint="x",
    ).cid
    ctx = ResearchContext(cite=cite, settings=settings)
    tool = WriteReportTool(DataAccess([], settings=settings), ctx=ctx)
    tool.coordinator = FakeCoordinator()

    result = asyncio.run(
        tool.run(
            topic="测试报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[{"heading": "章节", "body": f"正文 {{cite:{cid}}}", "cids": [cid]}],
            risks=["风险一"],
        )
    )

    assert result.ok is True, result.error
    review = result.metadata["review"]
    assert review["status"] == REVIEW_DONE
    assert review["topic"] == "测试报告"
    assert review["review_path"] and Path(review["review_path"]).is_file()
