"""Image links for local paths must stay valid Markdown when the path has spaces."""

import base64

from finharness.report.markdown import image_markdown

# A 1x1 transparent PNG, enough for python-docx to embed.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def test_plain_path_is_left_bare():
    assert image_markdown("标题", "/tmp/charts/x.png") == "![标题](/tmp/charts/x.png)"


def test_path_with_spaces_is_angle_bracket_wrapped():
    rendered = image_markdown("增速", r"F:\python project\FinCopilot\output\charts\a b.png")
    assert rendered == r"![增速](<F:\python project\FinCopilot\output\charts\a b.png>)"


def test_wrapped_destination_does_not_break_on_angle_bracket_in_name():
    rendered = image_markdown("t", "/tmp/a > b.png")
    assert rendered.startswith("![t](<")
    assert rendered.endswith(">)")
    assert "\\>" in rendered


def test_docx_export_embeds_image_when_path_has_spaces(tmp_path):
    """The angle brackets are Markdown syntax and must not reach the filesystem."""
    from docx import Document

    chart = tmp_path / "output dir" / "增长 图.png"
    chart.parent.mkdir(parents=True)
    chart.write_bytes(_PNG)

    out = tmp_path / "report.docx"
    from finharness.report.docx_export import export_markdown_to_docx

    export_markdown_to_docx(image_markdown("图表", str(chart)), out_path=out)

    document = Document(str(out))
    assert len(document.inline_shapes) == 1, "the chart must be embedded, not noted as missing"
