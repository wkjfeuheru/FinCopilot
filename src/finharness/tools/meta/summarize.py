"""summarize_document：为长文档准备可派发的分片索引（docs 03.4 · 03.10）。

一次性把整篇长文交给模型是行不通的：它超过窗口，也超过单条工具结果的预算。本工具
用结构感知切分产出带稳定 id 的分片清单，**不在内部派生子代理**。主 Agent 发现
``spawn_agent`` 后按片消化，只把带 ``[P01]`` 的结论收回主上下文。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from finharness.context.tokens import default_counter
from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.pdf_fetch import extract_pdf_pages
from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, Tier, ToolGroup, tool
from finharness.shared.summarize import document_outline, split_document
from finharness.tools.base import BaseTool
from finharness.workspace import Workspace

_RESULT_TOKENS = 4000
_DIRECT_LIMIT_TOKENS = 3000
_TEXT_MAX_BYTES = 2_000_000


class SummarizeInput(BaseModel):
    """``summarize_document`` 的输入模型（作为 ``params_model`` 整体使用）。

    "text 与 path 二选一"是**跨字段**约束，因此它仍由一个手写模型承载。
    """

    text: str | None = Field(default=None, description="待切分的文档正文；与 path 二选一")
    path: str | None = Field(
        default=None,
        description="待切分文件路径（PDF 或文本，限 output/ 与 data_cache/ 目录内）",
    )
    question: str | None = Field(
        default=None, description="聚焦问题，可选；写入每片建议任务句"
    )
    title: str | None = Field(default=None, description="文档标题，可选；用于结果抬头")

    @model_validator(mode="after")
    def _require_a_source(self) -> SummarizeInput:
        if not (self.text and self.text.strip()) and not (self.path and self.path.strip()):
            raise ValueError("必须提供 text 或 path 之一")
        return self


@tool(
    name="summarize_document",
    description=(
        "对一份长文档（研报/公告/PDF）做结构感知切分，返回带 id 的分片索引，"
        "不把原文拉入主上下文。可用 path 指向已落盘的本地文件，或直接给 text。"
        "拿到索引后按片 spawn_agent 消化（每片一条任务，结论以 [P01] 开头），"
        "或用 read_pdf 按页精读。需要长文档摘要时先调本工具。"
    ),
    capability=Capability.META,
    tier=Tier.LAZY,
    group=ToolGroup.META,
    timeout=30,
    result_tokens=_RESULT_TOKENS,
    output_schema_note=(
        "返回分片索引：每片 id、标签、页码或路径、建议任务句；不含分片全文。"
    ),
    params_model=SummarizeInput,
)
class SummarizeDocumentTool(BaseTool):
    async def _dispatch(
        self,
        *,
        text: str | None = None,
        path: str | None = None,
        question: str | None = None,
        title: str | None = None,
    ) -> RawData:
        body, source_path, pages = await asyncio.to_thread(
            self._resolve_source, text=text, path=path
        )
        if not body.strip():
            raise ValueError("文档内容为空，无法切分")
        resolved_title = title or self._title_from(body, source_path)
        counter = default_counter()
        total_tokens = counter.count(body).tokens
        if total_tokens <= _DIRECT_LIMIT_TOKENS:
            chunks = split_document(body, counter=counter) or []
            if len(chunks) != 1:
                from finharness.shared.summarize import Chunk, chunk_id

                chunks = [
                    Chunk(id=chunk_id(1), seq=1, text=body, label=self._title_from(body, source_path))
                ]
            note = "文档较短，未分片，可直接 spawn 一条或 read_file / read_pdf。"
        else:
            chunks = split_document(body, counter=counter)
            if not chunks:
                raise ValueError("文档无法切分为可摘要的分片")
            note = f"共 {len(chunks)} 片。按片 spawn_agent，主上下文只汇合带 [P01] 的结论。"
        outline = document_outline(body)
        ranges = _page_ranges(body, chunks, pages)
        content = _render_index(
            title=resolved_title,
            note=note,
            chunks=chunks,
            ranges=ranges,
            source_path=source_path,
            question=question,
            outline=outline,
        )
        return RawData(
            kind="text",
            text=content,
            endpoint="meta:summarize_document",
            params={
                "title": resolved_title,
                "chunks": len(chunks),
                "path": source_path,
                "question": question,
            },
        )

    def _resolve_source(
        self, *, text: str | None, path: str | None
    ) -> tuple[str, str | None, list[str] | None]:
        """返回 ``(正文, 源文件路径, 逐页文本)``；上限保证读取有界。"""
        if text and text.strip():
            return text, None, None
        target = Workspace(self.data.settings).resolve_read(path or "")
        if not target.is_file():
            raise ValueError(f"文件不存在：{path}")
        if target.suffix.lower() == ".pdf":
            try:
                data = target.read_bytes()
            except OSError as exc:
                raise ValueError(f"读取 PDF 失败：{exc}") from exc
            try:
                pages = extract_pdf_pages(data)
            except AdapterError as exc:
                raise ValueError(str(exc)) from exc
            return "\n".join(pages), str(target), pages
        raw = target.read_bytes()[:_TEXT_MAX_BYTES]
        return raw.decode("utf-8", "replace"), str(target), None

    @staticmethod
    def _title_from(body: str, source_path: str | None) -> str:
        first = next((line.strip() for line in body.splitlines() if line.strip()), "")
        if first:
            return first[:60]
        return Path(source_path).stem if source_path else "文档"


def _page_ranges(body: str, chunks, pages: list[str] | None) -> list[tuple[int, int] | None]:
    if not pages:
        return [None] * len(chunks)
    starts: list[int] = []
    cursor = 0
    for page in pages:
        starts.append(cursor)
        cursor += len(page) + 1

    def page_at(pos: int) -> int:
        for index, _start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else cursor
            if pos < end:
                return index + 1
        return len(pages)

    ranges: list[tuple[int, int] | None] = []
    search_from = 0
    for chunk in chunks:
        needle = (chunk.text[len(chunk.overlap) :] if chunk.overlap else chunk.text).strip()
        needle = needle[:80]
        idx = body.find(needle, search_from) if needle else -1
        if idx < 0:
            idx = body.find(chunk.text[:40], search_from)
        if idx < 0:
            ranges.append(None)
            continue
        start_page = page_at(idx)
        end_page = page_at(idx + max(len(chunk.text) - 1, 0))
        ranges.append((start_page, end_page))
        search_from = idx + 1
    return ranges


def _render_index(
    *,
    title: str,
    note: str,
    chunks,
    ranges: list[tuple[int, int] | None],
    source_path: str | None,
    question: str | None,
    outline: str,
) -> str:
    lines = [f"# {title}", note, ""]
    if source_path:
        lines.append(f"源文件：{source_path}")
        lines.append("")
    if outline:
        lines.append("文档骨架：")
        lines.append(outline[:800])
        lines.append("")
    lines.append("分片索引（不含全文）。建议一次 spawn_agent，每片一条任务，结论以该片 id 开头。")
    lines.append("")
    focus = question.strip() if question and question.strip() else "本片要点与关键数字"
    for chunk, span in zip(chunks, ranges):
        page_note = ""
        if span:
            start, end = span
            page_note = f"约第 {start} 页" if start == end else f"约第 {start}–{end} 页"
        loc = "；".join(part for part in (source_path, page_note) if part)
        lines.append(f"- [{chunk.id}] {chunk.label or '（无标题）'}" + (f"（{loc}）" if loc else ""))
        if source_path and span:
            start, end = span
            pages = str(start) if start == end else f"{start}-{end}"
            task = (
                f"读 {source_path} 的 {chunk.id}（{page_note}），"
                f'用 read_pdf pages="{pages}" 聚焦「{focus}」，结论以 {chunk.id} 开头。'
            )
        elif source_path:
            task = (
                f"读 {source_path} 的 {chunk.id}（{chunk.label}），"
                f"聚焦「{focus}」，结论以 {chunk.id} 开头。不要把原文整段交回。"
            )
        else:
            task = (
                f"消化分片 {chunk.id}（{chunk.label}），聚焦「{focus}」，"
                f"结论以 {chunk.id} 开头。不要把原文整段交回。"
            )
        lines.append(f"  建议任务：{task}")
    return "\n".join(lines).rstrip()
