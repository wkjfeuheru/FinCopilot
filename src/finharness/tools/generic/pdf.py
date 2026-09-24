"""read_pdf：按页读取本地 PDF（docs 03.4）。

长文档（研报、公告）的正文动辄数十页，整段塞进上下文既不可能也无必要。本工具给出
**分页**这一原始能力：抓取类工具把 PDF 落盘并交出一个路径，模型随后按需读取某几页。

这与 `read_file` 的分工是清楚的：`read_file` 读会话已指向的产物或缓存载荷，而
`read_pdf` 专门处理 PDF 的页寻址——它是"同一份材料，但要更细的一层"，因此属于
``GENERIC`` 组，复核者与 worker 子代理同样可以用它精读长材料。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.pdf_fetch import read_pdf_pages
from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, Tier, ToolGroup, param, tool
from finharness.shared.fencing import (
    EXTERNAL_NOTICE,
    RESULT_CLOSE,
    fence,
    neutralize,
)
from finharness.tools.base import BaseTool
from finharness.workspace import Workspace

# 默认返回页数：一页足以为读者提供上下文，又不至于让首次读取就撑满结果预算。
_DEFAULT_PAGE_SPAN = 1
# 单次最多返回的页数。它与结果预算相互独立：预算限制的是 token，这里限制的是页数，
# 使"读一份 300 页文档"不会因为页数而变成一次无界的抓取。
_MAX_PAGES_PER_READ = 10


@tool(
    name="read_pdf",
    description=(
        "按页读取本地 PDF（研报/公告正文）。给出 path，可选 pages（如 '2-5'）。"
        "用于在拿到落盘句柄后精读指定页，避免一次性拉入整份长文档。"
    ),
    capability=Capability.FILE,
    # 只有在拿到落盘句柄（研报/公告 PDF）之后才有用，故与抓取类工具同属按需注入。
    tier=Tier.LAZY,
    group=ToolGroup.GENERIC,
    timeout=60,
    output_schema_note="返回所请求页的文本，逐页分段；越界时说明实际页范围。",
)
class ReadPdfTool(BaseTool):
    @param("path", desc="本地 PDF 路径（限 output/ 与 data_cache/ 目录内）")
    @param("pages", desc="页码范围，如 '3'、'1-3'、'2-'；省略时只读第 1 页")
    async def _dispatch(self, *, path: str, pages: str | None = None) -> RawData:
        target = Workspace(self.data.settings).resolve_read(path)
        try:
            # PDF 解析既 CPU 密集又阻塞；放进线程才能让工具超时真正生效，
            # 且阻塞期间不占住事件循环（多租户下这是跨租户 DoS 的另一条路径）。
            page_texts, first, last, total = await asyncio.to_thread(
                read_pdf_pages, target, pages
            )
        except AdapterError as exc:
            raise ValueError(str(exc)) from exc

        requested = (pages or "").strip()
        if not requested and total > _DEFAULT_PAGE_SPAN:
            last = min(_DEFAULT_PAGE_SPAN, total)
            page_texts = page_texts[:last]
            first = 1

        if last - first + 1 > _MAX_PAGES_PER_READ:
            last = first + _MAX_PAGES_PER_READ - 1
            page_texts = page_texts[: _MAX_PAGES_PER_READ]

        text = self._render(target, first, last, total, page_texts)
        return RawData(
            kind="pdf_text",
            text=text,
            paths=[str(target)],
            endpoint="pdf:read",
            params={"path": str(target), "pages": f"{first}-{last}", "total": total},
        )

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        return (raw.text or "（未读取到内容）"), [raw]

    @staticmethod
    def _render(target: Path, first: int, last: int, total: int, pages: list[str]) -> str:
        """逐页分段渲染，并交代本次读的是哪一段。"""
        lines = [EXTERNAL_NOTICE, "", f"文件：{target}（共 {total} 页；本次读取第 {first}-{last} 页）", ""]
        for offset, page in enumerate(pages):
            number = first + offset
            lines.append(fence(number, target))
            lines.append(f"第 {number} 页")
            lines.append(neutralize(page.strip()) or "（本页无可抽取文本，可能是图片/扫描页）")
            lines.append(RESULT_CLOSE)
            lines.append("")
        if last < total:
            lines.append(f"（尚有 {total - last} 页未读取；用 pages=\"{last + 1}-\" 继续）")
        return "\n".join(lines).rstrip()
