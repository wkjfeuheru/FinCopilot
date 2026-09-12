"""write_report: turn a structured outline into a report artefact (docs 3.9).

Deliberately takes no ``path`` argument. The output location is derived from the
topic, which has two consequences: the naming convention cannot be bypassed, and
— because the permission gate only skips confirmation for writes carrying a
path inside the artefact directories — producing a report always asks the user
first.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.report.pipeline import (
    ReportOutline,
    ReportPipeline,
    ReportSection,
    ReportValidationError,
)
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup


class SectionInput(BaseModel):
    heading: str = Field(description="章节标题")
    body: str = Field(description="章节正文（markdown，结论性数字需带 {cite:cid}）")
    cids: list[str] = Field(default_factory=list, description="本章节引用的 citation id")
    charts: list[str] = Field(default_factory=list, description="本章节引用的图表路径（make_chart 产出）")


class ReportInput(BaseModel):
    topic: str = Field(description="报告主题，用于文件命名")
    core_view: list[str] = Field(description="核心观点，每条须含 {cite:cid}")
    sections: list[SectionInput] = Field(description="正文章节")
    risks: list[str] = Field(description="风险提示，建议 ≥5 条")
    formats: list[str] = Field(
        default_factory=lambda: ["md", "docx"],
        description="产出形态，默认同时产出 md 与 docx",
    )


class WriteReportTool(BaseTool):
    name = "write_report"
    description = (
        "把结构化大纲渲染成研报（markdown + docx）。正文中的结论性数字必须带引用，"
        "未标注来源的数字会在报告中被标记。"
    )
    input_model = ReportInput
    permission = PermissionLevel.WRITE
    group = ToolGroup.FIN_OUTPUT
    timeout = 120
    output_schema_note = "产出 output/<topic>_<YYYYMMDD>.md 与 .docx。"

    async def _dispatch(
        self,
        *,
        topic: str,
        core_view: list[str],
        sections: list[dict],
        risks: list[str],
        formats: list[str] | None = None,
    ) -> RawData:
        outline = ReportOutline(
            topic=topic,
            core_view=list(core_view),
            sections=[
                ReportSection(
                    heading=str(item.get("heading", "")),
                    body=str(item.get("body", "")),
                    cids=[str(c) for c in item.get("cids") or []],
                    charts=[str(c) for c in item.get("charts") or []],
                )
                for item in sections
            ],
            risks=list(risks),
        )
        pipeline = ReportPipeline(cite=self.ctx.cite, settings=self.data.settings)
        try:
            artifact = pipeline.export(outline, formats=tuple(formats or ("md", "docx")))
        except ReportValidationError as exc:
            # Precise, actionable feedback so the model can fix the outline.
            raise ValueError("报告校验未通过：" + "；".join(exc.problems)) from exc

        attachments = [artifact.markdown_path]
        if artifact.docx_path:
            attachments.append(artifact.docx_path)

        lines = [f"已生成报告：{artifact.topic}"]
        lines.append(f"- markdown：{artifact.markdown_path}")
        if artifact.docx_path:
            lines.append(f"- docx：{artifact.docx_path}")
        if artifact.docx_error:
            lines.append(f"- docx 导出失败（已保留 markdown）：{artifact.docx_error}")
        if artifact.citations:
            lines.append(f"- 引用数据 {len(artifact.citations)} 条")
        if artifact.warnings:
            lines.append("- 提示：")
            lines.extend(f"  - {w}" for w in artifact.warnings)

        return RawData(
            kind="text",
            text="\n".join(lines),
            paths=attachments,
            endpoint="report:pipeline",
            params={"topic": artifact.topic, "citations": len(artifact.citations)},
        )

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """Report results carry file attachments for the transport layer."""
        return (raw.text or "（报告生成失败）"), [raw]
