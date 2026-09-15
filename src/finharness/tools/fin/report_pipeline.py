"""研报流水线：校验大纲、组装 markdown、导出 docx（docs 3.9）。

模型提供结构化内容；本模块负责格式、引用编号与无来源数字检查，从而让每份
报告具有一致的形态与一致的可溯源保证。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from finharness.config.settings import Settings
from finharness.data.citation import CitationRegistry
from finharness.tools.fin.docx_export import DISCLAIMER, DocxExportError, export_markdown_to_docx
from finharness.utils.markdown import image_markdown

CITE_RE = re.compile(r"\{cite:(cit_\d+)\}")
# 带单位或百分号的数字才是必须可溯源的形态；裸年份与章节编号被有意排除。
# 英文数量级缩写（pct/pp/bp）也计作单位：若不如此，写成 "10.3pct" 的差值会
# 带着一个未标注来源的数字径直绕过溯源检查。
UNSOURCED_NUMBER_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:%|％|元|亿元|万元|倍|股|天|次|个百分点|pct|pp|bp))",
    re.IGNORECASE,
)
UNSOURCED_MARK = "[!无来源:{n}]"
# 不得出现在报告正文中的「内部取数」词汇。工具名与接口名属于系统管线；
# 附录已承载溯源信息，正文再复述会显得像机器输出。检测针对本会话引用中
# 实际出现的名称，而非硬编码列表。
TOOL_NAME_RE = re.compile(r"`?([a-z][a-z0-9_]+_[a-z0-9_]+)`?")
ENDPOINT_HINT_RE = re.compile(r"接口\s*[`“\"']?([A-Za-z0-9_.:]+)")


class ReportValidationError(ValueError):
    """当大纲无法产出有效报告时抛出。"""

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
    """文件系统安全的主题名：非法字符替换为下划线。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", topic.strip())
    return cleaned.strip("_")[:60] or "report"


class ReportPipeline:
    """大纲 -> 校验后的 markdown -> docx，并附带引用与警告。"""

    def __init__(
        self,
        *,
        cite: CitationRegistry,
        settings: Settings,
        tool_names: set[str] | None = None,
    ) -> None:
        self.cite = cite
        self.settings = settings
        # 完整工具目录，由 write_report 在能访问注册表时注入：正文不得提及任何
        # 内部工具，包括本会话从未调用的（"本可用 X 但未调用"读起来同样如此）。
        self.tool_names = tool_names or set()

    # -- 校验 ------------------------------------------------------------------
    def validate(self, outline: ReportOutline) -> None:
        """校验大纲：主题、核心观点引用、章节正文与引用、风险提示等。

        校验失败时抛出 ReportValidationError，其 problems 逐一说明问题。
        """
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

    # -- 组装 ------------------------------------------------------------------
    def _internal_name_warnings(self, text: str, warnings: list[str]) -> None:
        """标记暴露系统内部取数管线的正文文本。

        这里不适宜硬失败——就像无来源数字检查一样，读者仍需要这份报告——
        因此改为警告，供作者与审阅者处置。
        """
        known_tools = {item.tool for item in self.cite.all()} | self.tool_names
        known_interfaces: set[str] = set()
        for item in self.cite.all():
            endpoint = item.endpoint or ""
            if ":" in endpoint:
                known_interfaces.add(endpoint.rsplit(":", 1)[1])
        if not known_tools:
            return
        for match in TOOL_NAME_RE.finditer(text):
            name = match.group(1)
            if name in known_tools:
                warnings.append(f"正文暴露内部工具名：{name}（溯源由引用附录承载）")
        for match in ENDPOINT_HINT_RE.finditer(text):
            name = match.group(1)
            if name in known_interfaces:
                warnings.append(f"正文暴露内部接口名：{name}（溯源由引用附录承载）")

    def build_markdown(self, outline: ReportOutline) -> tuple[str, list[str], list[str]]:
        """返回 (markdown, 按序排列的引用 id, 警告列表)。"""
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
        body_parts = [f"- {v}" for v in outline.core_view]
        body_parts.extend(s.body for s in outline.sections)
        body_parts.extend(outline.risks)
        self._internal_name_warnings("\n".join(body_parts), warnings)

        lines.append("## 核心观点")
        lines.append("")
        for view in outline.core_view:
            lines.append(f"- {inline_cites(view)}")
        lines.append("")

        for section in outline.sections:
            lines.append(f"## {section.heading}")
            lines.append("")
            body = inline_cites(section.body)
            # 章节上列出的 cids 是对溯源的显式声明；此处加以引用，使其即使
            # 正文未逐一写明每个 id，也会出现在附录中。
            trailing = " ".join(f"[{number_for(cid)}]" for cid in section.cids if cid)
            body = self._mark_unsourced(body, warnings)
            lines.append(body + (f" {trailing}" if trailing else ""))
            lines.append("")
            for chart in section.charts:
                if Path(chart).is_file():
                    lines.append(image_markdown(section.heading, chart))
                    lines.append("")
                else:
                    warnings.append(f"图表缺失：{chart}")
                    lines.append(f"[图片缺失：{chart}]")
                    lines.append("")

        lines.append("## 风险提示")
        lines.append("")
        for index, risk in enumerate(outline.risks, start=1):
            # 风险提示通常引用触发它的数字，因此与正文采用相同的引用编号与
            # 无来源数字处理。
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
        """标记未携带引用标记的段落中的数字。

        以报告而非阻断的方式处理：硬失败会被普通数字误触发，而读者仍需要
        报告的其他内容。
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
        """生成「数据来源」附录表格；无引用时返回占位文案。"""
        if not ordered:
            return "（本次报告未引用数据）"
        lines = ["| 编号 | 引用 |", "|---|---|"]
        for cid in ordered:
            citation = self.cite.get(cid)
            # to_markdown() 会输出列表项目符号；在表格单元格内该标记是噪声，
            # 故将其去掉。
            detail = citation.to_markdown().lstrip("- ") if citation else cid
            lines.append(f"| [{number_for_index(cid, ordered)}] | {detail} |")
        return "\n".join(lines)

    # -- 导出 ------------------------------------------------------------------
    def export(
        self, outline: ReportOutline, *, formats: tuple[str, ...] = ("md", "docx")
    ) -> ReportArtifact:
        """写出所请求的格式；markdown 始终先于 docx。"""
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
                # markdown 仍作为兜底产物保留。
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
