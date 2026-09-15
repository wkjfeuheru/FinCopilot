"""产物交付与 write_report 治理。

下载端点必须限制在产物目录之内，且生成报告必须经过用户确认——M6 的验收标准取决于
后者，而只有当 write_report 不接受 ``path`` 参数时该标准才成立。
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
from tests.server.conftest import authed_client


def _artifact_client(tmp_path):
    """认证过的客户端：产物按用户隔离，端点只服务 output/<user>/ 内的文件。"""
    settings = Settings(
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "cache" / "memory.db",
            "auth_db": tmp_path / "cache" / "users.db",
        },
        data={"cache_dir": tmp_path / "cache"},
    )
    return authed_client(TestClient(create_app(provider=None, settings=settings)))


def test_artifact_endpoint_refuses_paths_outside_the_artefact_dirs(tmp_path):
    client = _artifact_client(tmp_path)

    response = client.get("/v1/artifacts", params={"path": str(tmp_path / ".." / "secrets.txt")})

    assert response.status_code in (403, 404)


def test_artifact_endpoint_serves_a_file_inside_output(tmp_path):
    client = _artifact_client(tmp_path)
    # 产物写入该用户自己的 output/<user>/ 子目录。
    output = tmp_path / "output" / client.finharness_user["id"]
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "report.md"
    artifact.write_text("# hello", encoding="utf-8")

    response = client.get("/v1/artifacts", params={"path": str(artifact)})

    assert response.status_code == 200
    assert "hello" in response.text


def test_artifact_endpoint_rejects_a_missing_file(tmp_path):
    client = _artifact_client(tmp_path)

    response = client.get(
        "/v1/artifacts",
        params={"path": str(tmp_path / "output" / client.finharness_user["id"] / "nope.md")},
    )

    assert response.status_code == 404


def test_artifact_endpoint_does_not_serve_another_users_artifacts(tmp_path):
    """关键隔离约束：A 的产物路径不能由 B 下载。"""
    client = _artifact_client(tmp_path)
    other = tmp_path / "output" / "u_someone_else"
    other.mkdir(parents=True, exist_ok=True)
    artifact = other / "report.md"
    artifact.write_text("# 他人的报告", encoding="utf-8")

    response = client.get("/v1/artifacts", params={"path": str(artifact)})

    assert response.status_code == 403


# --- write_report 治理 -------------------------------------------------------

def test_write_report_takes_no_path_argument():
    """路径参数的缺失正是强制确认的原因；此处显式守护这一约束。"""
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

    assert decision.verdict is Verdict.DENY  # 用户已拒绝
    assert "拒绝" in decision.reason


def test_make_chart_is_read_only_so_it_skips_confirmation(tmp_path):
    """按文档 3.4.1，图表属于产物；无需审批。"""
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
    # 该消息必须指明字段名，以便模型修复大纲。
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


# --- 写入时的风险复核（文档 03.10）-------------------------------------------

class FakeCoordinator:
    """充当子代理的替身：记录请求，返回预置的评审意见。"""

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
    assert "## 附录" not in sent  # 引用表不属于复核材料


def test_review_file_is_written_next_to_the_report(tmp_path):
    coordinator = FakeCoordinator()
    tool, cid = make_writer(tmp_path, coordinator=coordinator)

    result = run_write(tool, cid)

    reviews = [p for p in result.attachments if p.endswith(".review.md")]
    assert len(reviews) == 1
    assert Path(reviews[0]).is_file()
    assert "数字无来源" in Path(reviews[0]).read_text(encoding="utf-8")


def test_write_report_survives_a_review_failure(tmp_path):
    """复核失败绝不能丢弃一份渲染完好的报告。"""
    coordinator = FakeCoordinator(raises=RuntimeError("provider down"))
    tool, cid = make_writer(tmp_path, coordinator=coordinator)

    result = run_write(tool, cid)

    assert result.ok is True, result.error
    assert "风险终审未完成" in result.content
    # 失败必须读作“未经复核”，而不是一条中性的说明。
    assert "未经独立复核" in result.content
    # 报告产物仍然随附。
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
    """重写同一主题时，不得对仍然有效的复核结果重复执行。"""
    first = FakeCoordinator()
    tool, cid = make_writer(tmp_path, coordinator=first)
    run_write(tool, cid)

    # 第二次调用：全新的 coordinator，同一主题，报告已复核过。
    second = FakeCoordinator()
    tool.coordinator = second
    result = run_write(tool, cid)

    assert second.calls == []
    assert "复用" in result.content


def test_a_revised_report_is_reviewed_again(tmp_path):
    """该锁存基于内容：正文一旦变化，就必须重新复核。"""
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
