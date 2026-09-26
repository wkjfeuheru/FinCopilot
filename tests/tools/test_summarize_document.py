"""summarize_document：产出可派发的分片索引，不再内部 spawn（docs 03.4 · 03.10）。"""

from __future__ import annotations

import asyncio
import re

from finharness.config.settings import ContextSettings, Settings
from finharness.data.access import DataAccess
from finharness.tools.meta.summarize import SummarizeDocumentTool


def make_tool(tmp_path) -> SummarizeDocumentTool:
    settings = Settings(
        context=ContextSettings(),
        paths={"output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )
    return SummarizeDocumentTool(DataAccess([], settings=settings))


def long_document(sections: int = 6) -> str:
    parts = ["某券商行业深度报告：2026年展望"]
    for index in range(1, sections + 1):
        parts.append(f"{index}.{index} 第{index}章 关键议题")
        parts.append(f"本章要点与数据 {'细节' * 300}")
    return "\n".join(parts)


def run(coro):
    return asyncio.run(coro)


def test_a_short_document_returns_a_single_chunk_index(tmp_path):
    tool = make_tool(tmp_path)

    result = run(tool.run(text="很短的文档。"))

    assert result.ok is True, result.error
    assert "- [P01]" in result.content
    assert result.content.count("- [P01]") == 1
    assert "细节" * 50 not in result.content


def test_a_long_document_returns_an_index_without_full_text(tmp_path):
    tool = make_tool(tmp_path)
    body = long_document()

    result = run(tool.run(text=body, question="产能过剩怎么看"))

    assert result.ok is True, result.error
    ids = re.findall(r"- \[(P\d{2})\]", result.content)
    assert ids[0] == "P01"
    assert len(set(ids)) >= 2
    assert body not in result.content
    assert "spawn_agent" in result.content
    assert "产能过剩怎么看" in result.content


def test_chunk_ids_are_listed_in_document_order(tmp_path):
    tool = make_tool(tmp_path)

    result = run(tool.run(text=long_document()))

    ids = re.findall(r"- \[(P\d{2})\]", result.content)
    assert ids == sorted(set(ids), key=lambda value: int(value[1:]))


def test_it_requires_a_source(tmp_path):
    tool = make_tool(tmp_path)

    result = run(tool.run())

    assert result.ok is False
    assert "text" in (result.error or "") or "path" in (result.error or "")


def test_a_local_pdf_index_includes_the_path_and_pages(tmp_path):
    from tests.data.pdf_fixtures import make_multi_page_pdf

    directory = tmp_path / "cache" / "pdf"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "r.pdf"
    target.write_bytes(make_multi_page_pdf(["zqxalpha", "zqxbeta"]))

    tool = make_tool(tmp_path)
    result = run(tool.run(path=str(target), question="核心结论"))

    assert result.ok is True, result.error
    assert str(target) in result.content
    assert "P01" in result.content
    assert "read_pdf" in result.content or "页" in result.content


def test_it_is_lazy_and_read_only_without_a_coordinator(tmp_path):
    from finharness.shared.declaration import Tier
    from finharness.tools.base import PermissionLevel, ToolGroup

    assert SummarizeDocumentTool.permission is PermissionLevel.READ
    assert SummarizeDocumentTool.group is ToolGroup.META
    assert SummarizeDocumentTool.needs_coordinator is False
    assert SummarizeDocumentTool.timeout == 30
    assert SummarizeDocumentTool.tier is Tier.LAZY


def test_search_for_long_document_hits_summarize_and_spawn(tmp_path):
    from finharness.config.settings import ToolSettings
    from finharness.tools.registry import ToolRegistry

    settings = Settings(
        tools=ToolSettings(),
        data={"cache_dir": tmp_path / "cache"},
    )
    registry = ToolRegistry(DataAccess([], settings=settings), settings=settings)
    names = [brief.name for brief in registry.search("长文档摘要", limit=10)]
    assert "summarize_document" in names
    assert "spawn_agent" in names
