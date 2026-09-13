"""Artefact delivery and write_report governance.

The download endpoint must stay inside the artefact directories, and producing a
report must pass through user confirmation — M6's acceptance criterion depends on
the latter, and it only holds while write_report takes no ``path`` argument.
"""

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.permissions.gate import PermissionGate
from finharness.server.api import create_app
from finharness.tools.fin.writer import ReportInput, WriteReportTool


def test_artifact_endpoint_refuses_paths_outside_the_artefact_dirs(tmp_path):
    settings = Settings(
        paths={"output_dir": tmp_path / "output"}, data={"cache_dir": tmp_path / "cache"}
    )
    client = TestClient(create_app(provider=None, settings=settings))

    response = client.get("/v1/artifacts", params={"path": str(tmp_path / ".." / "secrets.txt")})

    assert response.status_code in (403, 404)


def test_artifact_endpoint_serves_a_file_inside_output(tmp_path):
    settings = Settings(
        paths={"output_dir": tmp_path / "output"}, data={"cache_dir": tmp_path / "cache"}
    )
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "report.md"
    artifact.write_text("# hello", encoding="utf-8")
    client = TestClient(create_app(provider=None, settings=settings))

    response = client.get("/v1/artifacts", params={"path": str(artifact)})

    assert response.status_code == 200
    assert "hello" in response.text


def test_artifact_endpoint_rejects_a_missing_file(tmp_path):
    settings = Settings(
        paths={"output_dir": tmp_path / "output"}, data={"cache_dir": tmp_path / "cache"}
    )
    client = TestClient(create_app(provider=None, settings=settings))

    response = client.get("/v1/artifacts", params={"path": str(tmp_path / "output" / "nope.md")})

    assert response.status_code == 404


# --- write_report governance -------------------------------------------------

def test_write_report_takes_no_path_argument():
    """The absence of a path is what forces confirmation; guard it explicitly."""
    fields = set(ReportInput.model_fields)
    assert "path" not in fields
    assert {"topic", "core_view", "sections", "risks"} <= fields


def test_write_report_must_be_confirmed_because_it_has_no_whitelisted_path(tmp_path):
    settings = Settings(
        permission={"default_mode": "default"},
        paths={"output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )
    asked: list[str] = []

    async def confirm(name, args):
        asked.append(name)
        return False

    gate = PermissionGate(settings=settings, confirm=confirm)
    tool = WriteReportTool(DataAccess([], settings=settings), ctx=ResearchContext(
        cite=CitationRegistry(), settings=settings
    ))

    decision = asyncio.run(gate.check(tool, {"topic": "t"}))

    assert asked == ["write_report"]
    from finharness.permissions.modes import Verdict

    assert decision.verdict is Verdict.DENY  # user declined
    assert "拒绝" in decision.reason


def test_make_chart_is_read_only_so_it_skips_confirmation(tmp_path):
    """Charts are artefacts per docs 3.4.1; they need no approval."""
    from finharness.tools.fin.chart import MakeChartTool
    from finharness.tools.base import PermissionLevel

    assert MakeChartTool.permission is PermissionLevel.READ


def test_write_report_returns_both_artefact_paths(tmp_path):
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

    result = asyncio.run(
        tool.run(
            topic="测试报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[{"heading": "章节", "body": f"正文 {{cite:{cid}}}", "cids": [cid]}],
            risks=["风险一"],
        )
    )

    assert result.ok is True, result.error
    assert len(result.attachments) == 2
    assert all(Path(p).is_file() for p in result.attachments)
    assert any(p.endswith(".md") for p in result.attachments)
    assert any(p.endswith(".docx") for p in result.attachments)


def test_write_report_reports_validation_problems_to_the_model(tmp_path):
    settings = Settings(
        paths={"output_dir": tmp_path / "output"}, data={"cache_dir": tmp_path / "cache"}
    )
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    tool = WriteReportTool(DataAccess([], settings=settings), ctx=ctx)

    result = asyncio.run(
        tool.run(topic="", core_view=["没有引用"], sections=[], risks=[])
    )

    assert result.ok is False
    assert "报告校验未通过" in result.error
    # The message must name the fields, so the model can repair the outline.
    assert "topic" in result.error


def test_write_report_warns_when_the_body_exposes_tool_names(tmp_path):
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

    result = asyncio.run(
        tool.run(
            topic="测试报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[{
                "heading": "说明",
                "body": f"get_quote 返回的行情与标的不符，故改用 get_kline {{cite:{cid}}}",
                "cids": [cid],
            }],
            risks=["风险一"],
        )
    )

    assert result.ok is True, result.error
    assert "正文暴露内部工具名" in result.content


# --- risk review on write (docs 03.10) ---------------------------------------

class FakeCoordinator:
    """Stands in for the sub-agent: records the request, returns canned comments."""

    def __init__(self, *, summary="### [高] 数字无来源\n- 位置：核心观点", ok=True,
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


def make_writer(tmp_path, *, coordinator) -> tuple[WriteReportTool, str]:
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
    tool.coordinator = coordinator
    return tool, cid


def run_write(tool, cid):
    return asyncio.run(
        tool.run(
            topic="测试报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[{"heading": "章节", "body": f"正文 {{cite:{cid}}}", "cids": [cid]}],
            risks=["风险一"],
        )
    )


def test_write_report_runs_the_risk_review_and_returns_its_comments(tmp_path):
    coordinator = FakeCoordinator()
    tool, cid = make_writer(tmp_path, coordinator=coordinator)

    result = run_write(tool, cid)

    assert result.ok is True, result.error
    assert len(coordinator.calls) == 1
    assert "风险终审意见" in result.content
    assert "数字无来源" in result.content


def test_review_receives_the_report_body_without_the_appendix(tmp_path):
    coordinator = FakeCoordinator()
    tool, cid = make_writer(tmp_path, coordinator=coordinator)

    run_write(tool, cid)

    sent = coordinator.calls[0]["markdown"]
    assert "测试报告" in sent
    assert "## 附录" not in sent  # the citation table is not review material


def test_review_file_is_written_next_to_the_report(tmp_path):
    coordinator = FakeCoordinator()
    tool, cid = make_writer(tmp_path, coordinator=coordinator)

    result = run_write(tool, cid)

    reviews = [p for p in result.attachments if p.endswith(".review.md")]
    assert len(reviews) == 1
    assert Path(reviews[0]).is_file()
    assert "数字无来源" in Path(reviews[0]).read_text(encoding="utf-8")


def test_write_report_survives_a_review_failure(tmp_path):
    """A failed review must never discard a perfectly rendered report."""
    coordinator = FakeCoordinator(raises=RuntimeError("provider down"))
    tool, cid = make_writer(tmp_path, coordinator=coordinator)

    result = run_write(tool, cid)

    assert result.ok is True, result.error
    assert "风险终审未完成" in result.content
    # Failure must read as "unreviewed", not as a neutral note.
    assert "未经独立复核" in result.content
    # The report artefacts are still attached.
    assert any(p.endswith(".md") for p in result.attachments)
    assert any(p.endswith(".docx") for p in result.attachments)


def test_write_report_survives_a_declined_review(tmp_path):
    coordinator = FakeCoordinator(ok=False, error="max_turns_exhausted")
    tool, cid = make_writer(tmp_path, coordinator=coordinator)

    result = run_write(tool, cid)

    assert result.ok is True
    assert "风险终审未完成" in result.content
    assert "max_turns_exhausted" in result.content
    assert "未经独立复核" in result.content


def test_a_current_review_is_reused_instead_of_paying_twice(tmp_path):
    """Rewriting the same topic must not re-run a review that is still current."""
    first = FakeCoordinator()
    tool, cid = make_writer(tmp_path, coordinator=first)
    run_write(tool, cid)

    # Second call: a fresh coordinator, same topic, report already reviewed.
    second = FakeCoordinator()
    tool.coordinator = second
    result = run_write(tool, cid)

    assert second.calls == []
    assert "复用" in result.content


def test_a_revised_report_is_reviewed_again(tmp_path):
    """The latch is content-based: a changed body must earn a fresh review."""
    first = FakeCoordinator()
    tool, cid = make_writer(tmp_path, coordinator=first)
    run_write(tool, cid)

    second = FakeCoordinator()
    tool.coordinator = second
    result = asyncio.run(
        tool.run(
            topic="测试报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[{"heading": "章节", "body": f"修订后的正文 {{cite:{cid}}}", "cids": [cid]}],
            risks=["风险一", "风险二"],
        )
    )

    assert result.ok is True
    assert len(second.calls) == 1


def test_write_report_without_a_coordinator_still_works(tmp_path):
    tool, cid = make_writer(tmp_path, coordinator=None)

    result = run_write(tool, cid)

    assert result.ok is True
    assert "风险终审" not in result.content
