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
