"""summarize_document：长文档的 map-reduce 摘要（docs 03.4 · 03.10）。

一次性把整篇长文交给模型是行不通的：它超过窗口，也超过单条工具结果的预算。本工具
把文档切成结构感知的分片，**复用会话的子代理**为每片生成摘要（map），再归并成一份
全局摘要（reduce）。全过程只在主上下文留下最终摘要与句柄，中间材料不回流。

三项设计取舍：

* **复用 spawn，而不是自己调模型。** 工具拿不到 provider（这是刻意的隔离），但循环
  会把会话的协调器注入进来。因此 map/reduce 走既有的 ``Coordinator.spawn``——同一套
  子代理、可观测性与用量记账。
* **id 是契约，顺序不是。** 子代理并发返回、分片超过一批还要分批，所以每片带稳定
  id，归并前按 id 重排（见 ``coordinator/summarize.py``）。丢失的片如实标注缺失。
* **摘要之外保留原文。** 摘要必然损失细节，所以结果里给出分片明细与源文件路径，
  需要核对原话时可用 ``read_pdf`` / ``read_file`` 下钻。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from finharness.coordinator import GENERAL_FOCUS
from finharness.coordinator.numbers import (
    CLIP_MARKER,
    find_gaps,
    page_span,
    pages_for_anchor,
    repair_task,
)
from finharness.coordinator.summarize import (
    MapResult,
    batch_chunks,
    document_outline,
    map_task,
    needs_layered_reduce,
    order_by_id,
    parse_map_summary,
    reduce_groups,
    reduce_task,
    split_document,
)
from finharness.context.tokens import default_counter
from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.pdf_fetch import extract_pdf_pages
from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, Tier, ToolGroup, tool
from finharness.tools.generic.fencing import EXTERNAL_NOTICE
from finharness.tools.generic.files import _resolve_within

# 结果声明一个较宽的预算：它要容纳一份全局摘要 + 分片索引 + 路径，但仍是有界的。
_RESULT_TOKENS = 4000
# 低于此 token 数的文档不值得走 map：直接一次归并更省、也更连贯。
_DIRECT_LIMIT_TOKENS = 3000
# 模型返回的摘要正文上限，防止单片的"摘要"本身膨胀成第二篇长文。但研报分片的关键
# 数字密集，预算过紧会正好切在数字中间（如"同比少增…"），而归并只看得到截断后的
# 文本，被切掉的数字再也找不回——因此留足余量，宁可略长也要保住数字。
_MAP_SUMMARY_TOKENS = 1000
_TEXT_MAX_BYTES = 2_000_000
# 数字核对阶段最多派发的补齐任务数：每个缺口一次按页回读，封顶以免异常输入放大成本。
_MAX_REPAIR_TASKS = 4


class SummarizeInput(BaseModel):
    """``summarize_document`` 的输入模型（作为 ``params_model`` 整体使用）。

    "text 与 path 二选一"是**跨字段**约束，因此它仍由一个手写模型承载：``@param``
    声明的是单字段描述，而这条规则约束的是两个字段的联合取值。
    """

    text: str | None = Field(default=None, description="待摘要的文档正文；与 path 二选一")
    path: str | None = Field(
        default=None,
        description="待摘要文件路径（PDF 或文本，限 output/ 与 data_cache/ 目录内）",
    )
    question: str | None = Field(
        default=None, description="聚焦问题，可选；给出时围绕它组织摘要"
    )
    title: str | None = Field(default=None, description="文档标题，可选；用于结果抬头")

    @model_validator(mode="after")
    def _require_a_source(self) -> "SummarizeInput":
        if not (self.text and self.text.strip()) and not (self.path and self.path.strip()):
            raise ValueError("必须提供 text 或 path 之一")
        return self


@tool(
    name="summarize_document",
    description=(
        "对一份长文档（研报/公告/PDF）做 map-reduce 摘要：分片并行摘要后归并成全局摘要，"
        "全程不把原文拉入主上下文。可用 path 指向已落盘的本地文件，或直接给 text。"
        "需要核对原话时改用 read_pdf 精读。"
    ),
    capability=Capability.META,
    # map-reduce 会派发多个子代理，成本高，只在确需消化长文时才注入。
    tier=Tier.LAZY,
    group=ToolGroup.META,
    # 多批次的并发子代理，每组都要跑完自己的若干轮，因此预算覆盖整批；再叠加一轮
    # 数字核对的按页回读。
    timeout=360,
    result_tokens=_RESULT_TOKENS,
    # 由循环注入：工具无法自行构建协调器（那需要 provider，而工具看不到它）。
    needs_coordinator=True,
    output_schema_note=(
        "返回全局摘要（带分片出处）、数字核对（缺值处及按页回读结果）、分片索引与落盘路径。"
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
        if self.coordinator is None:
            raise ValueError("当前会话未启用子代理协调器，无法执行 map-reduce 摘要")
        # 读取与 PDF 抽取都会阻塞（研报动辄数十页），放进线程使工具超时可真正生效，
        # 阻塞期间不占住事件循环。
        body, source_path, pages = await asyncio.to_thread(
            self._resolve_source, text=text, path=path
        )
        if not body.strip():
            raise ValueError("文档内容为空，无法摘要")
        resolved_title = title or self._title_from(body, source_path)

        counter = default_counter()
        total_tokens = counter.count(body).tokens
        # 短文档不走 map：分片再归并在小文本上只会损失连贯性并多花一次调用。
        if total_tokens <= _DIRECT_LIMIT_TOKENS:
            summary = await self._direct(body, question=question)
            detail_path = self._save_detail(resolved_title, body, [])
            verify = await self._verify_numbers(
                summary=summary, maps=[], pages=pages, source_path=source_path
            )
            return self._result(
                resolved_title, summary, [], source_path, detail_path,
                total=1, note="（文档较短，未分片，直接归并）", verify=verify,
            )

        chunks = split_document(body, counter=counter)
        if not chunks:
            raise ValueError("文档无法切分为可摘要的分片")
        outline = document_outline(body)

        maps = await self._map(chunks, outline=outline, question=question, counter=counter)
        ordered = order_by_id(maps, total=len(chunks))
        missing = [item.chunk.id for item in ordered if not item.ok]
        summary = await self._reduce(ordered, question=question, counter=counter)
        # 归并只看得到（可能已被裁剪的）分片摘要，因此缺值要在归并之后再核一遍。
        verify = await self._verify_numbers(
            summary=summary, maps=ordered, pages=pages, source_path=source_path
        )

        detail_path = self._save_detail(
            resolved_title, body, [item for item in ordered]
        )
        note = f"分片缺漏：{'、'.join(missing)}" if missing else ""
        return self._result(
            resolved_title, summary, ordered, source_path, detail_path,
            total=len(chunks), note=note, verify=verify,
        )

    # -- 输入解析 --------------------------------------------------------------
    def _resolve_source(
        self, *, text: str | None, path: str | None
    ) -> tuple[str, str | None, list[str] | None]:
        """返回 ``(正文, 源文件路径, 逐页文本)``；上限保证读取有界。

        ``pages`` 仅 PDF 有值。数字核对要靠它把缺口的锚文本落回具体页码，因此必须在
        抽取的同一处保留页边界，而不是事后重读一次文件（那样会抽两遍）。
        """
        if text and text.strip():
            return text, None, None
        settings = self.data.settings
        roots = [
            Path(settings.paths.output_dir).resolve(),
            (Path(settings.data.cache_dir) / "pdf").resolve(),
            (Path(settings.data.cache_dir) / "parquet").resolve(),
        ]
        target = _resolve_within(path or "", roots)
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

    # -- 数字核对 --------------------------------------------------------------
    async def _verify_numbers(
        self,
        *,
        summary: str,
        maps: list[MapResult],
        pages: list[str] | None,
        source_path: str | None,
    ) -> list[str]:
        """归并之后的数字核对：找出缺值处，并按页派子代理回读补齐。

        只判**缺值**（裁剪标记、量词后的省略号），不比对原文数字全集——一份研报有
        数百个数字，摘要本就只收关键项，按全集比对会把正常取舍成批误报为缺失。扫描
        范围含各分片摘要：裁剪标记正是留在那里，只看最终摘要会漏。

        这是 best-effort：定位不到、子代理未完成、来源不是 PDF，都只如实记录为
        "未补齐/需人工核对"，绝不让一次成功的摘要因此失败。
        """
        haystack = "\n".join([*(item.summary for item in maps if item.ok), summary])
        gaps = find_gaps(haystack)
        if not gaps:
            return []

        located = [
            (gap, pages_for_anchor(pages, gap.anchor) if pages else []) for gap in gaps
        ]
        findings: dict[str, str] = {}
        can_read = bool(pages and source_path)
        if can_read:
            repairable = [(gap, hits) for gap, hits in located if hits][:_MAX_REPAIR_TASKS]
            tasks = [
                repair_task(path=source_path, pages=hits, cue=gap.cue)
                for gap, hits in repairable
            ]
            if tasks:
                outcomes = await self._spawn_repairs(tasks)
                for (gap, _), outcome in zip(repairable, outcomes):
                    if outcome.ok and (outcome.summary or "").strip():
                        findings[gap.anchor] = outcome.summary.strip()

        if not pages:
            note = "（来源非 PDF，无法按页回读，以下缺口需人工核对）"
        elif not source_path:
            note = "（未留下源文件路径，无法按页回读）"
        else:
            note = ""
        return self._render_verify(located, findings, note=note, can_read=can_read)

    async def _spawn_repairs(self, tasks: list[str]):
        """派子代理按页回读。失败返回空列表而非抛出——补齐是 best-effort。"""
        try:
            return await self.coordinator.spawn(tasks=tasks, focus=GENERAL_FOCUS)
        except Exception:  # noqa: BLE001 - 补齐失败不得推翻已成功的摘要
            return []

    @staticmethod
    def _render_verify(
        located, findings: dict[str, str], *, note: str, can_read: bool
    ) -> list[str]:
        """把核对结果渲染成结果文本的一段。"""
        lines = ["", "数字核对：以下位置摘要提到了量，但未给出具体数值。"]
        if note:
            lines.append(note)
        for gap, hits in located:
            where = "第 " + "、".join(str(page) for page in hits) + " 页" if hits else "未定位页码"
            found = findings.get(gap.anchor)
            if found:
                lines.append(f"  - 「{gap.cue}」（{where}）已按原文补齐：{found}")
            elif hits and can_read:
                lines.append(
                    f"  - 「{gap.cue}」（{where}）未能补齐，"
                    f'可用 read_pdf pages="{page_span(hits)}" 精读该页'
                )
            else:
                lines.append(f"  - 「{gap.cue}」（{where}）需人工核对")
        return lines

    # -- map / reduce ---------------------------------------------------------
    async def _map(
        self, chunks, *, outline: str, question: str | None, counter
    ) -> list[MapResult]:
        """逐批并发地取得每个分片的摘要。"""
        results: list[MapResult] = []
        for group in batch_chunks(chunks):
            tasks = [
                map_task(chunk, outline=outline, question=question) for chunk in group
            ]
            outcomes = await self.coordinator.spawn(
                tasks=tasks, focus=GENERAL_FOCUS
            )
            for chunk, outcome in zip(group, outcomes):
                if not outcome.ok:
                    results.append(
                        MapResult(
                            chunk=chunk,
                            ok=False,
                            error=outcome.error or "子代理未完成",
                        )
                    )
                    continue
                _, parsed = parse_map_summary(outcome.summary or "", expected_id=chunk.id)
                results.append(
                    MapResult(
                        chunk=chunk,
                        summary=self._clip(parsed, counter, _MAP_SUMMARY_TOKENS),
                    )
                )
        return results

    async def _direct(self, body: str, *, question: str | None) -> str:
        """短文档：一次归并即可，无需分片。"""
        task = reduce_task(
            [f"（未分片，全文如下）\n{body}"],
            total=1,
            question=question,
            final=True,
        )
        outcomes = await self.coordinator.spawn(tasks=[task], focus=GENERAL_FOCUS)
        if not outcomes or not outcomes[0].ok:
            error = outcomes[0].error if outcomes else "子代理未返回结果"
            raise ValueError(f"摘要未能生成：{error}")
        return outcomes[0].summary or ""

    async def _reduce(self, ordered: list[MapResult], *, question: str | None, counter) -> str:
        """分层归并：材料过大时先分组归并，再做全局归并。"""
        usable = [item for item in ordered if item.ok and item.summary.strip()]
        if not usable:
            raise ValueError("所有分片均摘要失败，无法归并")
        items = [f"[{item.chunk.id}] {item.summary}" for item in usable]
        note = ""
        if any(not item.ok for item in ordered):
            failed = "、".join(item.chunk.id for item in ordered if not item.ok)
            note = f"（以下分片摘要失败，内容缺失：{failed}）"

        if needs_layered_reduce(items, counter=counter):
            groups = reduce_groups(items)
            tasks = [
                reduce_task(group, total=len(ordered), question=question, final=False)
                for group in groups
            ]
            outcomes = await self.coordinator.spawn(tasks=tasks, focus=GENERAL_FOCUS)
            merged = [
                outcome.summary
                for outcome in outcomes
                if outcome.ok and (outcome.summary or "").strip()
            ]
            if not merged:
                raise ValueError("分组归并未返回可用结果")
            items = [f"（阶段性归并）\n{text}" for text in merged]

        final_task = reduce_task(items, total=len(ordered), question=question, final=True)
        outcomes = await self.coordinator.spawn(tasks=[final_task], focus=GENERAL_FOCUS)
        if not outcomes or not outcomes[0].ok:
            error = outcomes[0].error if outcomes else "子代理未返回结果"
            raise ValueError(f"归并未能生成：{error}")
        return (outcomes[0].summary or "") + ("\n" + note if note else "")

    # -- 输出 ------------------------------------------------------------------
    def _result(
        self,
        title: str,
        summary: str,
        ordered: list[MapResult],
        source_path: str | None,
        detail_path: str | None,
        *,
        total: int,
        note: str = "",
        verify: list[str] | None = None,
    ) -> RawData:
        """把摘要、核对结果、分片索引与句柄组装成有界文本。"""
        lines = [EXTERNAL_NOTICE, "", f"# 文档摘要：{title}", ""]
        if note:
            lines.append(note)
            lines.append("")
        lines.append(summary.strip() or "（摘要为空）")
        if verify:
            lines.extend(verify)
        lines.append("")
        lines.append(f"分片数：{total}")
        if ordered:
            lines.append("分片索引：")
            for item in ordered:
                state = "✓" if item.ok else "✗（摘要失败）"
                lines.append(f"  - [{item.chunk.id}] {state} {item.chunk.label or ''}".rstrip())
        if source_path:
            lines.append(f"原文档：{source_path}")
        if detail_path:
            lines.append(f"分片明细：{detail_path}")
        lines.append("（需要核对原话时，用 read_pdf / read_file 按下钻路径精读）")
        return RawData(
            kind="text",
            text="\n".join(lines).rstrip(),
            paths=[p for p in (source_path, detail_path) if p],
            endpoint="meta:summarize_document",
            params={"chunks": total, "title": title},
        )

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        return (raw.text or "（摘要生成失败）"), [raw]

    def _save_detail(
        self, title: str, body: str, ordered: list[MapResult]
    ) -> str | None:
        """把全文与逐片摘要写到 ``output/`` 下，作为可回查的句柄。

        摘要必然有损，因此必须留下下钻路径：调用方据此核对某句话的原文，而不是只能
        相信摘要。写入失败只是没有句柄，不应让整个摘要失败。
        """
        from datetime import datetime

        try:
            safe = re.sub(r"[^\w\-]", "_", title)[:40] or "doc"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = Path(self.data.settings.paths.output_dir) / "summaries"
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{safe}_{stamp}.md"
            parts = [f"# {title}", "", "## 分片摘要", ""]
            for item in ordered:
                state = "" if item.ok else "（摘要失败）"
                parts.append(f"### [{item.chunk.id}] {item.chunk.label} {state}".rstrip())
                parts.append(item.summary or item.error or "（无内容）")
                parts.append("")
            parts.extend(["## 全文", "", body])
            target.write_text("\n".join(parts), encoding="utf-8")
            return str(target)
        except OSError:
            return None

    @staticmethod
    def _clip(text: str, counter, max_tokens: int) -> str:
        from finharness.context.tokens import truncate_to_tokens

        return truncate_to_tokens(text, counter, max_tokens, marker=CLIP_MARKER)
