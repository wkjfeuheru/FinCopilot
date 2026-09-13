"""write_report: turn a structured outline into a report artefact (docs 3.9).

Deliberately takes no ``path`` argument. The output location is derived from the
topic, which has two consequences: the naming convention cannot be bypassed, and
— because the permission gate only skips confirmation for writes carrying a
path inside the artefact directories — producing a report always asks the user
first.

Writing a report also runs the risk-review sub-agent (docs 03.10). That is the
only trigger for review: putting it here, rather than in a tool the model may
choose, makes "every report gets reviewed" a property of the system instead of a
hope about model behaviour.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.report.pipeline import (
    ReportOutline,
    ReportPipeline,
    ReportSection,
    ReportValidationError,
)
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup

# The appendix is a citation table, not prose: the reviewer judges the body and
# the risk section, so sending the table would cost tokens without changing a
# single comment.
APPENDIX_MARKER = "\n## 附录"
# A review records the digest of the body it judged, so an unchanged rewrite
# reuses the verdict instead of paying for the same review twice.
REVIEW_DIGEST_PREFIX = "<!-- reviewed-body:"


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
    # Report rendering plus one review turn against a real provider. The default
    # 30s budget cannot cover the provider round trip a review needs.
    timeout = 300
    output_schema_note = "产出 output/<topic>_<YYYYMMDD>.md 与 .docx，并附风险终审意见。"
    needs_coordinator = True

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
        pipeline = ReportPipeline(
            cite=self.ctx.cite,
            settings=self.data.settings,
            # The catalogue doubles as the detection dictionary: a body that
            # names any internal tool — called this session or not — is
            # exposing system plumbing instead of speaking to the reader.
            tool_names=set(self.registry.names()) if self.registry is not None else set(),
        )
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

        review_path = await self._review_risk(artifact, lines)
        if review_path:
            attachments.append(review_path)

        return RawData(
            kind="text",
            text="\n".join(lines),
            paths=attachments,
            endpoint="report:pipeline",
            params={"topic": artifact.topic, "citations": len(artifact.citations)},
        )

    async def _review_risk(self, artifact, lines: list[str]) -> str | None:
        """Run the risk reviewer and fold its verdict into the result text.

        Returns the review file path when one was written. The report itself must
        survive any review failure: ``BaseTool.run`` turns an exception here into
        a blanket ``ok=False``, which would discard a report that was rendered
        perfectly well. So every failure degrades to a warning line.
        """
        if self.coordinator is None:
            return None
        markdown_path = Path(artifact.markdown_path)
        review_path = markdown_path.with_suffix(".review.md")

        body = self._report_body(markdown_path)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

        # Latch on content, not on time. Rewriting the same topic overwrites the
        # report, so an mtime comparison would call every review stale; comparing
        # the reviewed body's digest reuses the verdict when nothing changed and
        # re-reviews only when the report was actually revised.
        if self._reviewed_digest(review_path) == digest:
            lines.append(f"- 风险终审：已完成（复用 {review_path}）")
            return str(review_path)

        try:
            result = await self.coordinator.review_risk(topic=artifact.topic, markdown=body)
        except Exception as exc:  # noqa: BLE001 - never fail a rendered report
            lines.append(self._unreviewed_warning(f"{type(exc).__name__}: {exc}"))
            return None

        if not result.ok:
            lines.append(self._unreviewed_warning(result.error))
            return None

        comments = (result.summary or "").strip()
        lines.append("- 风险终审意见（独立复核，供你决定是否修订）：")
        lines.extend(f"  {line}" for line in (comments or "未发现实质性问题").splitlines())
        lines.append(f"- 完整意见：{review_path}")
        try:
            review_path.write_text(
                f"{REVIEW_DIGEST_PREFIX}{digest} -->\n\n"
                f"# 风险终审意见：{artifact.topic}\n\n{comments}\n",
                encoding="utf-8",
            )
        except OSError as exc:
            # Losing the sidecar file is not worth failing the report over; the
            # comments are already in the result text.
            lines.append(f"- 终审意见落盘失败：{exc}")
            return None
        return str(review_path)

    @staticmethod
    def _unreviewed_warning(error: str | None) -> str:
        """A failure to review must read as "unreviewed", not as a neutral note.

        The report body is deliberately left untouched (docs 03.10.5), so the tool
        result is the only place a caller learns the review did not happen. A bland
        "未完成" line is easy to skim past; this states the missing assurance.
        """
        detail = f"（{error}）" if error else ""
        return (
            f"- ⚠ 风险终审未完成{detail}：**本报告未经独立复核**，"
            "不得视为已复核交付；如需复核请重新成稿触发，或人工核对关键数字。"
        )

    @staticmethod
    def _report_body(markdown_path: Path) -> str:
        """The report without its citation appendix, for the reviewer to read."""
        text = markdown_path.read_text(encoding="utf-8")
        marker = text.find(APPENDIX_MARKER)
        return text if marker == -1 else text[:marker]

    @staticmethod
    def _reviewed_digest(review_path: Path) -> str | None:
        """The body digest a previous review was based on, if any is recorded."""
        try:
            head = review_path.read_text(encoding="utf-8")[:200]
        except OSError:
            return None
        marker = REVIEW_DIGEST_PREFIX
        start = head.find(marker)
        if start == -1:
            return None
        remainder = head[start + len(marker):]
        return remainder.split(" ", 1)[0].strip() or None

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """Report results carry file attachments for the transport layer."""
        return (raw.text or "（报告生成失败）"), [raw]
