"""summarize_document：map-reduce 编排、id 复原、失败如实降级（docs 03.4 · 03.10）。

用一个脚本化的 stub 协调器替代真子代理：被测对象是工具自身的编排——分片、分批、
按 id 重排、分层归并、以及某片失败时不伪装完整。真正的子代理机制由
``tests/coordinator`` 覆盖。
"""

from __future__ import annotations

import asyncio
import re

import pytest

from finharness.config.settings import ContextSettings, Settings
from finharness.coordinator.reviewer import SubAgentResult
from finharness.data.access import DataAccess
from finharness.tools.meta.summarize import SummarizeDocumentTool


class StubCoordinator:
    """按任务文本回放摘要：把要求的 id 原样回显，其余用固定文案。

    它模拟真实子代理的两个关键行为：map 的输出可能**乱序**返回，以及某些任务会
    失败。``shuffle_maps`` 打开时故意打乱返回顺序，用来证明工具是**按 id**而非按
    返回次序恢复全文顺序的。

    ``clip_text`` 不为空时，在每片摘要末尾追加该文本与截断标记，用来模拟"分片摘要被
    裁剪"；``repair_reply`` 是补齐任务（要求 read_pdf 的那类任务）的回话。
    """

    def __init__(
        self,
        *,
        shuffle_maps: bool = False,
        fail_ids: set[str] | None = None,
        clip_text: str | None = None,
        repair_reply: str = "第 2 页：同比少增 8600 亿元。",
    ):
        self.shuffle_maps = shuffle_maps
        self.fail_ids = fail_ids or set()
        self.clip_text = clip_text
        self.repair_reply = repair_reply
        self.calls: list[list[str]] = []

    async def spawn(self, *, tasks, focus="general", context=None):
        self.calls.append(list(tasks))
        results = []
        for task in tasks:
            # 补齐任务要求读 PDF 取数：按 repair_reply 回话，不套用摘要口吻。
            if "read_pdf" in task:
                results.append(
                    SubAgentResult(focus=focus, summary=self.repair_reply, task=task, ok=True)
                )
                continue
            identifier = self._id_in(task)
            if identifier and identifier in self.fail_ids:
                results.append(
                    SubAgentResult(
                        focus=focus, summary="", task=task, ok=False, error="子代理未完成"
                    )
                )
                continue
            # 摘要正文以 id 开头，模拟受约束的输出格式。
            summary = f"{identifier or 'P??'}\n本片要点：{identifier or '无'}。"
            if self.clip_text:
                summary += self.clip_text + "…（本片摘要已截断）"
            results.append(SubAgentResult(focus=focus, summary=summary, task=task, ok=True))
        if self.shuffle_maps and len(results) > 1:
            results = list(reversed(results))
        return results

    @staticmethod
    def _id_in(task: str) -> str | None:
        match = re.search(r"\b(P\d{2,3})\b", task)
        return match.group(1) if match else None


def make_tool(tmp_path, coordinator) -> SummarizeDocumentTool:
    settings = Settings(
        context=ContextSettings(),
        paths={"output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = SummarizeDocumentTool(DataAccess([], settings=settings))
    tool.coordinator = coordinator
    return tool


def long_document(sections: int = 6) -> str:
    """构造一份足够长、带明确章节标题的文档。"""
    parts = ["某券商行业深度报告：2026年展望"]
    for index in range(1, sections + 1):
        parts.append(f"{index}.{index} 第{index}章 关键议题")
        parts.append(f"本章要点与数据 {'细节' * 300}")
    return "\n".join(parts)


def run(coro):
    return asyncio.run(coro)


def test_a_short_document_skips_the_map_stage(tmp_path):
    """短文档不值得分片：只应发生一次归并调用。"""
    coordinator = StubCoordinator()
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(text="很短的文档。"))

    assert result.ok is True, result.error
    assert "未分片" in result.content
    # 只有一次 spawn（归并），没有 map 批次。
    assert len(coordinator.calls) == 1


def test_a_long_document_is_mapped_then_reduced(tmp_path):
    coordinator = StubCoordinator()
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(text=long_document()))

    assert result.ok is True, result.error
    assert "分片数" in result.content
    # map 批次 + 至少一次归并。
    assert len(coordinator.calls) >= 2
    # 每片都带出处 id，使读者可回查。
    assert re.search(r"\[P\d{2}\]", result.content)


def test_shuffled_map_results_are_restored_to_document_order(tmp_path):
    """并行乱序的返回必须按 id 复原，否则归并会读到错位的正文顺序。"""
    coordinator = StubCoordinator(shuffle_maps=True)
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(text=long_document()))

    assert result.ok is True, result.error
    # 索引按 id 递增列出，与返回次序无关。
    ids = re.findall(r"\[(P\d{2})\]", result.content)
    ordered = [item for item in ids if item in {f"P{index:02d}" for index in range(1, 30)}]
    assert ordered == sorted(set(ordered), key=lambda value: int(value[1:]))


def test_a_failed_chunk_is_reported_as_missing_not_hidden(tmp_path):
    """某片失败时必须如实标注缺失，而不是给出一份看似完整的摘要。"""
    coordinator = StubCoordinator(fail_ids={"P02"})
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(text=long_document()))

    assert result.ok is True, result.error
    assert "P02" in result.content
    assert "失败" in result.content


def test_more_chunks_than_one_batch_are_dispatched_in_batches(tmp_path):
    """分片数超过单批上限时自动分批，而不是一次派发超过上限的任务。"""
    from finharness.coordinator.summarize import MAX_TASKS_PER_BATCH

    coordinator = StubCoordinator()
    tool = make_tool(tmp_path, coordinator)
    # 章节多到产生 10 片以上。
    result = run(tool.run(text=long_document(sections=40)))

    assert result.ok is True, result.error
    map_batches = [
        call for call in coordinator.calls if len(call) <= MAX_TASKS_PER_BATCH
    ]
    assert map_batches
    total = int(re.search(r"分片数：(\d+)", result.content).group(1))
    assert total > MAX_TASKS_PER_BATCH


def test_the_full_text_and_chunk_detail_are_written_as_a_handle(tmp_path):
    """摘要必然有损，因此必须留下可下钻的句柄。"""
    from pathlib import Path

    coordinator = StubCoordinator()
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(text=long_document(), title="测试研报"))

    detail = re.search(r"分片明细：(.+)", result.content)
    assert detail, result.content
    path = Path(detail.group(1).strip())
    assert path.is_file()
    written = path.read_text(encoding="utf-8")
    assert "## 分片摘要" in written
    assert "## 全文" in written


def test_an_unwired_coordinator_fails_clearly(tmp_path):
    """未注入协调器时给出可理解的错误，而不是一个 None 引发的崩溃。"""
    settings = Settings(
        context=ContextSettings(),
        paths={"output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = SummarizeDocumentTool(DataAccess([], settings=settings))

    result = run(tool.run(text=long_document()))

    assert result.ok is False
    assert "协调器" in (result.error or "")


def test_it_requires_a_source(tmp_path):
    tool = make_tool(tmp_path, StubCoordinator())

    result = run(tool.run())

    assert result.ok is False
    assert "text" in (result.error or "") or "path" in (result.error or "")


def test_a_local_pdf_can_be_summarized_by_path(tmp_path):
    """接受本地 PDF 路径：与研报抓取的交界正是这个句柄。"""
    from tests.data.pdf_fixtures import make_multi_page_pdf

    directory = tmp_path / "cache" / "pdf"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "r.pdf"
    target.write_bytes(make_multi_page_pdf(["zqxalpha", "zqxbeta"]))

    coordinator = StubCoordinator()
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(path=str(target)))

    assert result.ok is True, result.error
    # 源文件路径如实回显，供后续下钻。
    assert str(target) in result.content


def test_it_is_lazy_and_read_only(tmp_path):
    from finharness.tools.base import PermissionLevel, ToolGroup
    from finharness.tools.declare import Tier

    assert SummarizeDocumentTool.permission is PermissionLevel.READ
    assert SummarizeDocumentTool.group is ToolGroup.META
    assert SummarizeDocumentTool.needs_coordinator is True
    assert SummarizeDocumentTool.tier is Tier.LAZY


# -- 数字核对（reduce 之后的按页回读） -----------------------------------------


def test_a_clipped_number_is_reported_as_a_gap(tmp_path):
    """缺值必须被如实指出，而不是让读者以为摘要已完整。"""
    coordinator = StubCoordinator(clip_text="同比少增")
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(text=long_document()))

    assert result.ok is True, result.error
    assert "数字核对" in result.content
    assert "同比少增" in result.content


def test_a_non_pdf_source_reports_the_gap_without_a_readback(tmp_path):
    """没有逐页原文就无从回读：只报告缺口，且不得派发 read_pdf 任务。"""
    coordinator = StubCoordinator(clip_text="同比少增")
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(text=long_document()))

    assert "需人工核对" in result.content
    assert "来源非 PDF" in result.content
    assert not any("read_pdf" in task for call in coordinator.calls for task in call)


def test_a_pdf_gap_is_located_and_read_back_by_page(tmp_path):
    """PDF 来源：定位到页码后派子代理用 read_pdf 回读，把缺的数字补回来。"""
    from tests.data.pdf_fixtures import make_multi_page_pdf

    directory = tmp_path / "cache" / "pdf"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "r.pdf"
    # 第 2 页含锚文本（ASCII，规避 PDF 字面量的编码问题），另外两页不含。
    target.write_bytes(make_multi_page_pdf(["zqxalpha", "VALUE-NEEDED 8600", "zqxbeta"]))

    coordinator = StubCoordinator(
        clip_text="VALUE-NEEDED 8600", repair_reply="第 2 页：同比少增 8600 亿元。"
    )
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(path=str(target)))

    assert result.ok is True, result.error
    assert "数字核对" in result.content
    # 缺值由子代理按页读回并落在结果里。
    assert "已按原文补齐" in result.content
    assert "8600 亿元" in result.content
    # 派出的补齐任务自包含：带源文件路径，并要求 read_pdf 读对页。
    repair_tasks = [task for call in coordinator.calls for task in call if "read_pdf" in task]
    assert repair_tasks
    assert str(target) in repair_tasks[0]
    assert 'pages="2"' in repair_tasks[0]


def test_a_gap_that_cannot_be_located_offers_the_page_handle(tmp_path):
    """定位不到页码时不得编一个页号，而应如实说未定位。"""
    coordinator = StubCoordinator(clip_text="同比少增")
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(text=long_document()))

    assert "未定位页码" in result.content


def test_a_failed_repair_does_not_fail_the_summary(tmp_path):
    """补齐是 best-effort：子代理未完成也只记为未补齐，摘要本身仍成功。"""
    from tests.data.pdf_fixtures import make_multi_page_pdf

    directory = tmp_path / "cache" / "pdf"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "r.pdf"
    target.write_bytes(make_multi_page_pdf(["VALUE-NEEDED 8600"]))

    class FailingRepairCoordinator(StubCoordinator):
        async def spawn(self, *, tasks, focus="general", context=None):
            if any("read_pdf" in task for task in tasks):
                raise RuntimeError("coordinator down")
            return await super().spawn(tasks=tasks, focus=focus, context=context)

    coordinator = FailingRepairCoordinator(clip_text="VALUE-NEEDED 8600")
    tool = make_tool(tmp_path, coordinator)

    result = run(tool.run(path=str(target)))

    assert result.ok is True, result.error
    assert "数字核对" in result.content
    assert "未能补齐" in result.content
