"""Small Markdown helpers shared by the report pipeline and chart tool."""

from __future__ import annotations


def image_markdown(alt: str, path: str) -> str:
    """Return a spec-compliant Markdown image link for ``path``.

    The CommonMark grammar forbids unescaped spaces in a bare link destination,
    so a Windows path like ``C:\\output dir\\chart.png`` would otherwise be
    parsed as plain text instead of an image. Such destinations are wrapped in
    ``<>`` — the escape hatch the spec provides — so viewers (including the web
    UI) render the image rather than printing the raw syntax.
    """
    destination = path.strip()
    if any(char.isspace() for char in destination):
        # A literal `>` would close the wrapper early; escape it.
        destination = destination.replace(">", "\\>")
        return f"![{alt}](<{destination}>)"
    return f"![{alt}]({destination})"
