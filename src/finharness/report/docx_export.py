"""Markdown -> docx export (docs 3.9.3).

Built from scratch with python-docx so styling is code (and therefore testable)
rather than a binary template that cannot be reviewed in a diff.
"""

from __future__ import annotations

import re
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.+?)\)\s*$")
TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|\s*$")

BODY_FONT = "SimSun"
HEADING_FONT = "SimHei"
MAX_IMAGE_INCHES = 6.0

DISCLAIMER = "本报告由 FinHarness 依据公开数据自动生成，仅供研究参考，不构成投资建议。"


class DocxExportError(RuntimeError):
    """Raised when the document cannot be produced."""


def _set_run_font(run, *, font: str) -> None:
    """Apply a font to both the ASCII and the East Asian script slots.

    python-docx only sets the ASCII font by default; without ``w:eastAsia`` Word
    falls back to a Latin face and Chinese glyphs render inconsistently.
    """
    from docx.oxml.ns import qn

    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def export_markdown_to_docx(markdown: str, *, out_path: str | Path, topic: str = "") -> Path:
    """Convert report markdown to ``.docx``; returns the written path."""
    try:
        from docx import Document
        from docx.shared import Inches
    except ImportError as exc:  # pragma: no cover - python-docx is a dependency
        raise DocxExportError(f"python-docx 不可用：{exc}") from exc

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()

    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()

        if not line.strip():
            index += 1
            continue

        # Tables: a header row, a separator row, then body rows.
        if TABLE_ROW_RE.match(line) and index + 1 < len(lines) and TABLE_SEP_RE.match(lines[index + 1]):
            block, index = _consume_table(lines, index)
            _write_table(document, block, font=BODY_FONT)
            continue

        image = IMAGE_RE.match(line)
        if image:
            caption, target = image.group(1), image.group(2).strip()
            # Paths containing spaces are angle-bracket wrapped for Markdown
            # validity; the brackets are syntax, not part of the filesystem path.
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1].replace("\\>", ">").strip()
            # Only local files are embedded; a missing one degrades to a note so
            # the rest of the document still exports.
            candidate = Path(target)
            if candidate.is_file():
                try:
                    document.add_picture(str(candidate), width=Inches(MAX_IMAGE_INCHES))
                except Exception as exc:  # noqa: BLE001 - bad image must not abort export
                    _add_note(document, f"[图片嵌入失败：{target}（{exc}）]", font=BODY_FONT)
                else:
                    if caption:
                        _add_caption(document, caption, font=BODY_FONT)
            else:
                _add_note(document, f"[图片缺失：{target}]", font=BODY_FONT)
            index += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            text = _strip_inline(heading.group(2))
            paragraph = document.add_heading(level=level)
            run = paragraph.add_run(text)
            _set_run_font(run, font=HEADING_FONT)
            index += 1
            continue

        paragraph = document.add_paragraph()
        run = paragraph.add_run(_strip_inline(line))
        _set_run_font(run, font=BODY_FONT)
        index += 1

    document.save(str(path))
    return path


def _consume_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    block: list[list[str]] = []
    index = start
    while index < len(lines) and TABLE_ROW_RE.match(lines[index]):
        if TABLE_SEP_RE.match(lines[index]):
            index += 1
            continue
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        block.append(cells)
        index += 1
    return block, index


def _write_table(document, block: list[list[str]], *, font: str) -> None:
    if not block:
        return
    width = max(len(row) for row in block)
    table = document.add_table(rows=0, cols=width)
    table.style = "Table Grid"
    for row in block:
        cells = table.add_row().cells
        for position in range(width):
            text = row[position] if position < len(row) else ""
            paragraph = cells[position].paragraphs[0]
            run = paragraph.add_run(_strip_inline(text))
            _set_run_font(run, font=font)


def _add_note(document, text: str, *, font: str) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = True
    _set_run_font(run, font=font)


def _add_caption(document, text: str, *, font: str) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.font.size = None  # inherit body size
    _set_run_font(run, font=font)


def _strip_inline(text: str) -> str:
    """Drop the markdown emphasis markers that docx does not need."""
    return text.replace("**", "").replace("`", "").strip()
