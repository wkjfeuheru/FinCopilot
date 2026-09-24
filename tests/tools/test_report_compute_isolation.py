"""write_report 的 docx 导出改走隔离计算通道时的契约。

隔离导出的意义是：把"嵌入图表、写二进制文档"这类重活移出主进程。但它必须
产出与进程内导出**等价**的文件——尤其是图表，worker 容器不挂载 output 目录，
若不随任务包一起送过去，隔离路径会静默退化成"图片缺失"，而进程内路径不会。
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

from finharness.compute.executor import ComputeResult
from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.tools.fin.writer import WriteReportTool


class _RecordingCompute:
    """记录被派发的任务，并返回一个可解压的 docx 载荷。"""

    def __init__(self, *, blob: bytes = b"PK\x03\x04fake-docx") -> None:
        self.calls: list[dict] = []
        self.blob = blob

    async def execute(self, *, kind: str, files, timeout_s: float = 120) -> ComputeResult:
        self.calls.append({"kind": kind, "files": dict(files), "timeout_s": timeout_s})
        return ComputeResult(
            "succeeded", blobs=("report.docx",),
            blob_data={"report.docx": base64.b64encode(self.blob).decode("ascii")},
        )


def _tool(tmp_path, compute) -> WriteReportTool:
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
    tool.compute = compute
    return tool, cid


def test_write_report_routes_docx_through_the_compute_channel(tmp_path):
    compute = _RecordingCompute()
    tool, cid = _tool(tmp_path, compute)

    result = asyncio.run(
        tool.run(
            topic="隔离报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[{"heading": "章节", "body": f"正文 {{cite:{cid}}}", "cids": [cid]}],
            risks=["风险一"],
        )
    )

    assert result.ok is True, result.error
    assert [call["kind"] for call in compute.calls] == ["docx_export"]
    assert compute.calls[0]["timeout_s"] > 0
    assert any(path.endswith(".md") for path in result.attachments)
    docx = [path for path in result.attachments if path.endswith(".docx")]
    assert docx and Path(docx[0]).read_bytes().startswith(b"PK")


def test_chart_images_travel_with_the_package_as_content_addressed_names(tmp_path):
    chart = tmp_path / "chart one.png"  # 含空格：走 Markdown 的 <> 包裹
    chart.write_bytes(b"\x89PNG-fake")

    class _ChartCompute(_RecordingCompute):
        async def execute(self, *, kind, files, timeout_s=120):
            self.calls.append({"kind": kind, "files": dict(files), "timeout_s": timeout_s})
            request = json.loads(files["request.json"])
            packaged = [name for name in files if name != "request.json"]
            assert len(packaged) == 1, "图表必须随任务包送达 worker"
            assert f"<{packaged[0]}>" in request["markdown"], "引用要改写成包内文件名"
            assert files[packaged[0]] == b"\x89PNG-fake"
            return ComputeResult(
                "succeeded", blobs=("report.docx",),
                blob_data={"report.docx": base64.b64encode(b"PK-docx").decode("ascii")},
            )

    compute = _ChartCompute()
    tool, cid = _tool(tmp_path, compute)

    result = asyncio.run(
        tool.run(
            topic="带图报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[
                {"heading": "章节", "body": f"正文 {{cite:{cid}}}", "cids": [cid],
                 "charts": [str(chart)]},
            ],
            risks=["风险一"],
        )
    )

    assert result.ok is True, result.error


def test_write_report_falls_back_in_process_when_no_compute_channel(tmp_path):
    """无 worker（本地/CLI）时仍产出 docx——声明的是"可以外包"不是"必须"。"""
    tool, cid = _tool(tmp_path, None)

    result = asyncio.run(
        tool.run(
            topic="本地报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[{"heading": "章节", "body": f"正文 {{cite:{cid}}}", "cids": [cid]}],
            risks=["风险一"],
        )
    )

    assert result.ok is True, result.error
    assert any(path.endswith(".docx") for path in result.attachments)


def test_failed_isolated_export_keeps_markdown_and_reports_the_error(tmp_path):
    class _FailingCompute:
        async def execute(self, *, kind, files, timeout_s=120):
            return ComputeResult("failed", error="worker_error:boom")

    tool, cid = _tool(tmp_path, _FailingCompute())

    result = asyncio.run(
        tool.run(
            topic="失败报告",
            core_view=[f"观点 {{cite:{cid}}}"],
            sections=[{"heading": "章节", "body": f"正文 {{cite:{cid}}}", "cids": [cid]}],
            risks=["风险一"],
        )
    )

    assert result.ok is True, result.error
    assert any(path.endswith(".md") for path in result.attachments)
    assert not any(path.endswith(".docx") for path in result.attachments)
    assert "docx" in result.content
