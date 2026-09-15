"""报告流水线与图表工具共用的小型 Markdown 辅助函数。"""

from __future__ import annotations


def image_markdown(alt: str, path: str) -> str:
    """为 ``path`` 返回符合规范的 Markdown 图片链接。

    CommonMark 语法禁止裸链接目标中出现未转义的空格，因此像
    ``C:\\output dir\\chart.png`` 这样的 Windows 路径会被解析为纯文本，
    而不是图片。此类目标会被包裹在 ``<>`` 中 —— 规范提供的转义方式 ——
    使查看器（包括 Web UI）渲染出图片，而不是打印原始语法。
    """
    destination = path.strip()
    if any(char.isspace() for char in destination):
        # 字面量 `>` 会提前闭合包裹；将其转义。
        destination = destination.replace(">", "\\>")
        return f"![{alt}](<{destination}>)"
    return f"![{alt}]({destination})"
