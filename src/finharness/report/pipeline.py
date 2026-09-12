"""Report pipeline: validate an outline, assemble markdown, export docx (docs 3.9).

The model supplies structured content; this module owns formatting, citation
numbering and the unsourced-number check, so every report has the same shape and
the same traceability guarantees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from finharness.config.settings import Settings
from finharness.data.citation import CitationRegistry
from finharness.report.docx_export import DISCLAIMER, DocxExportError, export_markdown_to_docx

PLACEHOLDER_RE = re.compile(r"\{(cite|chart|table|list):([^}]+)\}")
CITE_RE = re.compile(r"\{cite:(cit_\d+)\}")
# A number with a unit or percent sign is the shape that must be traceable;
# bare years and section numbers are deliberately excluded.
UNSOURCED_NUMBER_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:%|％|元|亿元|万元|倍|股|天|次|个百分点))"
)
UNSOURCED_MARK = "[!无来源:{n}]"


class ReportValidationError(ValueError):
    """Raised when an outline cannot produce a valid report."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("；".join(problems))


@dataclass(slots=True)
class ReportSection:
    heading: str
    body: str
    cids: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReportOutline:
    topic: str
    core_view: list[str] = field(default_factory=list)
    sections: list[ReportSection] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReportArtifact:
    topic: str
    markdown_path: str
    docx_path: str | None
    citations: list[str]
    warnings: list[str] = field(default_factory=list)
    docx_error: str | None = None


def safe_topic(topic: str) -> str:
    """Filesystem-safe topic: illegal characters become underscores."""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", topic.strip())
    return cleaned.strip("_")[:60] or "report"


class ReportPipeline:
    """Outline -> validated markdown -> docx, with citations and warnings."""

    def __init__(self, *, cite: CitationRegistry, settings: Settings) -> None:
        self.cite = cite
        self.settings = settings

    # -- validation -----------------------------------------------------------
    def validate(self, outline: ReportOutline) -> None:
        problems: list[str] = []
        if not outline.topic.strip():
            problems.append("topic 不能为空")
        if not outline.core_view:
            problems.append("core_view 至少需要一条核心观点")

        for position, view in enumerate(outline.core_view, start=1):
            if not CITE_RE.search(view):
                problems.append(f"核心观点第 {position} 条缺少引用标记 {{cite:...}}")
        if not outline.sections:
            problems.append("sections 至少需要一个章节")

        for position, section in enumerate(outline.sections, start=1):
            label = f"章节第 {position} 个「{section.heading or '未命名'}」"
            if not section.heading.strip():
                problems.append(f"{label} 缺少 heading")
            if not section.body.strip():
                problems.append(f"{label} 正文为空")
            if not section.cids and not CITE_RE.search(section.body):
                problems.append(f"{label} 缺少引用（cids 或正文中的 {{cite:...}}）")
            for chart in section.charts:
                if not Path(chart).is_file():
                    problems.append(f"{label} 引用的图表不存在：{chart}")
        if not outline.risks:
            problems.append("risks 至少需要一条风险提示")
        if problems:
            raise ReportValidationError(problems)

    # -- assembly -------------------------------------------------------------
    def build_markdown(self, outline: ReportOutline) -> tuple[str, list[str], list[str]]:
        """Return (markdown, citation ids in order, warnings)."""
        warnings: list[str] = []
        ordered: list[str] = []
        numbers: dict[str, int] = {}

        def number_for(cid: str) -> str:
            if cid not in numbers:
                numbers[cid] = len(numbers) + 1
                ordered.append(cid)
            return str(numbers[cid])

        def inline_cites(text: str) -> str:
            return CITE_RE.sub(
                lambda match: f"[{number_for(match.group(1))}]", text, count=0
            )

        lines: list[str] = [f"# {outline.topic}", ""]

        lines.append("## 核心观点")
        lines.append("")
        for view in outline.core_view:
            lines.append(f"- {inline_cites(view)}")
        lines.append("")

        for section in outline.sections:
            lines.append(f"## {section.heading}")
            lines.append("")
            body = inline_cites(section.body)
            # Cids listed on the section are an explicit claim of provenance;
            # reference them so they appear in the appendix even if the body did
            # not spell out every id.
            trailing = " ".join(f"[{number_for(cid)}]" for cid in section.cids if cid)
            body = self._mark_unsourced(body, warnings)
            lines.append(body + (f" {trailing}" if trailing else ""))
            lines.append("")
            for chart in section.charts:
                if Path(chart).is_file():
                    lines.append(f"![{section.heading}]({chart})")
                    lines.append("")
                else:
                    warnings.append(f"图表缺失：{chart}")
                    lines.append(f"[图片缺失：{chart}]")
                    lines.append("")

        lines.append("## 风险提示")
        lines.append("")
        for index, risk in enumerate(outline.risks, start=1):
            # Risks commonly cite the figures that trigger them, so they get the
            # same citation numbering and unsourced-number treatment as the body.
            rendered = inline_cites(risk)
            rendered = self._mark_unsourced(rendered, warnings)
            lines.append(f"{index}. {rendered}")
        lines.append("")

        lines.append("## 附录：数据来源")
        lines.append("")
        lines.append(self._appendix(ordered))
        lines.append("")
        lines.append(f"> {DISCLAIMER}")
        lines.append("")

        return "\n".join(lines), ordered, warnings

    def _mark_unsourced(self, text: str, warnings: list[str]) -> str:
        """Flag numbers in a paragraph that carries no citation marker.

        Reported rather than blocked: a hard failure would trip on ordinary
        figures, and the reader still needs the rest of the report.
        """
        marked: list[str] = []
        for paragraph in text.split("\n"):
            has_cite = "[" in paragraph and "]" in paragraph
            if has_cite or not paragraph.strip():
                marked.append(paragraph)
                continue
            counter = {"n": 0}

            def annotate(match: re.Match) -> str:
                counter["n"] += 1
                warnings.append(f"未标注来源的数字：{match.group(1)}")
                return match.group(1) + UNSOURCED_MARK.format(n=counter["n"])

            marked.append(UNSOURCED_NUMBER_RE.sub(annotate, paragraph))
        return "\n".join(marked)

    def _appendix(self, ordered: list[str]) -> str:
        if not ordered:
            return "（本次报告未引用数据）"
        lines = ["| 编号 | 引用 |", "|---|---|"]
        for cid in ordered:
            citation = self.cite.get(cid)
            # to_markdown() emits a list bullet; inside a table cell the marker
            # is noise, so strip it.
            detail = citation.to_markdown().lstrip("- ") if citation else cid
            lines.append(f"| [{number_for_index(cid, ordered)}] | {detail} |")
        return "\n".join(lines)

    # -- export ---------------------------------------------------------------
    def export(
        self, outline: ReportOutline, *, formats: tuple[str, ...] = ("md", "docx")
    ) -> ReportArtifact:
        """Write the requested formats; markdown always precedes docx."""
        self.validate(outline)
        markdown, ordered, warnings = self.build_markdown(outline)

        output_dir = Path(self.settings.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{safe_topic(outline.topic)}_{date.today().strftime('%Y%m%d')}"

        markdown_path = output_dir / f"{stem}.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        docx_path: str | None = None
        docx_error: str | None = None
        if "docx" in formats:
            target = output_dir / f"{stem}.docx"
            try:
                export_markdown_to_docx(markdown, out_path=target, topic=outline.topic)
            except (DocxExportError, OSError) as exc:
                # Markdown remains as the fallback artefact.
                docx_error = str(exc)
                warnings.append(f"docx 导出失败，已保留 markdown：{exc}")
            else:
                docx_path = str(target)

        return ReportArtifact(
            topic=outline.topic,
            markdown_path=str(markdown_path),
            docx_path=docx_path,
            citations=ordered,
            warnings=warnings,
            docx_error=docx_error,
        )


def number_for_index(cid: str, ordered: list[str]) -> int:
    return ordered.index(cid) + 1
