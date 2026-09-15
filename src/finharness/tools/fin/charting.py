"""共享的 matplotlib 配置：使用可渲染中文的字体，否则显式报错。

用缺少字形的字体渲染中文标签，会在保存的 PNG 中静默产生豆腐块，
这比直接失败更糟：读者无法分辨一张损坏的图表与一张空图表。
因此在开头就探测字体，若缺失则作为错误上报。
"""

from __future__ import annotations

import threading

# 优先级顺序覆盖 Windows、macOS 以及常见 Linux CJK 字体包。
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
    """当系统未安装可渲染中文的字体时抛出。"""


_lock = threading.Lock()
_resolved: str | None = None


def resolve_cjk_font() -> str:
    """返回可用的 CJK 字体族名称；若不可用则抛出 ``FontUnavailableError``。

    结果会被缓存：字体探测需要遍历系统字体列表，在一次会话中生成多张图表时，
    这一开销足以产生影响。
    """
    global _resolved
    with _lock:
        if _resolved is not None:
            return _resolved
        try:
            from matplotlib import font_manager
        except ImportError as exc:  # pragma: no cover - matplotlib 为必需依赖
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
    """应用已解析的字体与适用于无显示环境的后端。"""
    import matplotlib

    matplotlib.use("Agg", force=True)  # 无需显示器；图表以文件形式产出
    matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False  # 使用 CJK 字体时负号才能正常渲染
