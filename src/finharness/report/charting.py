"""Shared matplotlib setup: a Chinese-capable font or an explicit failure.

Rendering Chinese labels with a font that lacks the glyphs silently produces
tofu boxes in the saved PNG, which is worse than failing: the reader cannot tell
a broken chart from an empty one. So the font is probed up front and a miss is
reported as an error.
"""

from __future__ import annotations

import threading

# Preference order spans Windows, macOS and common Linux CJK packages.
CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "SimHei",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Heiti SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)


class FontUnavailableError(RuntimeError):
    """Raised when no Chinese-capable font is installed."""


_lock = threading.Lock()
_resolved: str | None = None


def resolve_cjk_font() -> str:
    """Return a usable CJK font family name, or raise ``FontUnavailableError``.

    Result is memoised: font discovery walks the system font list, which is slow
    enough to matter when several charts are produced in one session.
    """
    global _resolved
    with _lock:
        if _resolved is not None:
            return _resolved
        try:
            from matplotlib import font_manager
        except ImportError as exc:  # pragma: no cover - matplotlib is a dependency
            raise FontUnavailableError(f"matplotlib 不可用：{exc}") from exc

        available = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in CJK_FONT_CANDIDATES:
            if candidate in available:
                _resolved = candidate
                return candidate
        raise FontUnavailableError(
            "未找到可用的中文字体，无法渲染中文图表；"
            f"已尝试：{'、'.join(CJK_FONT_CANDIDATES)}。"
            "请安装其中任一中文字体（如 Noto Sans CJK SC）。"
        )


def apply_style(*, font: str) -> None:
    """Apply the resolved font and a headless-safe backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)  # no display needed; charts are files
    matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False  # minus sign renders with CJK fonts
