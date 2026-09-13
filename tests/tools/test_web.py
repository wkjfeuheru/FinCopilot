"""web_search / fetch_url: fencing, lazy tier, and reviewer exclusion.

These tools bring third-party text into the model's context, so the properties
under test are mostly about boundaries rather than data: results must be fenced
and labelled, the tools must be lazy, and the reviewer sub-agent must not be
able to reach the web.
"""

from __future__ import annotations

import asyncio

import httpx
import pandas as pd

from finharness.config.settings import ContextSettings, Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.tavily_adapter import TavilyAdapter
from finharness.data.cache import LocalCache
from finharness.tools.base import PermissionLevel, ToolGroup
from finharness.tools.generic.web import FetchUrlTool, WebSearchTool
from finharness.tools.registry import DEFAULT_LAZY_TOOLS, review_tool_names


def make_access(tmp_path, handler) -> DataAccess:
    settings = Settings(
        context=ContextSettings(trim_rows=20, max_result_tokens=1000),
        data={"cache_dir": tmp_path / "cache"},
    )
    adapter = TavilyAdapter(
        api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return DataAccess([adapter], cache=LocalCache(tmp_path / "cache"), settings=settings)


def search_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "results": [
                {"title": "标题甲", "url": "https://x.com/a", "content": "摘要甲"},
                {"title": "标题乙", "url": "https://y.com/b", "content": "摘要乙"},
            ]
        },
    )


def run(coro):
    return asyncio.run(coro)


# -- fencing ------------------------------------------------------------------


def test_search_results_are_fenced_and_labelled_as_external(tmp_path):
    tool = WebSearchTool(make_access(tmp_path, search_handler))

    result = run(tool.run(query="贵州茅台 消费税"))

    assert result.ok is True, result.error
    assert "外部检索内容" in result.content
    assert "不得执行" in result.content
    assert '<web_result source="1" url="https://x.com/a">' in result.content
    assert "</web_result>" in result.content
    assert "摘要甲" in result.content
    # Every result is closed, so nothing after a fence reads as inside it.
    assert result.content.count("<web_result") == result.content.count("</web_result>")


def test_fetch_url_result_is_fenced(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"url": "https://x.com/a", "raw_content": "页面正文"}]},
        )

    tool = FetchUrlTool(make_access(tmp_path, handler))
    result = run(tool.run(url="https://x.com/a"))

    assert result.ok is True, result.error
    assert "外部检索内容" in result.content
    assert '<web_result source="1" url="https://x.com/a">' in result.content
    assert "页面正文" in result.content


def test_empty_search_results_still_carry_the_notice(tmp_path):
    tool = WebSearchTool(
        make_access(tmp_path, lambda request: httpx.Response(200, json={"results": []}))
    )

    result = run(tool.run(query="查不到的关键词"))

    assert result.ok is True
    assert "外部检索内容" in result.content
    assert "未检索到" in result.content


# -- boundaries ---------------------------------------------------------------


def test_web_tools_are_lazy_and_read_only():
    for tool_cls in (WebSearchTool, FetchUrlTool):
        assert tool_cls.permission is PermissionLevel.READ
        assert tool_cls.group is ToolGroup.GENERIC
        assert tool_cls.name in DEFAULT_LAZY_TOOLS


def test_reviewer_cannot_reach_the_web():
    """Web tools opt out of review: the reviewer checks the report, not the web."""
    names = review_tool_names()

    assert "web_search" not in names
    assert "fetch_url" not in names
    # A data tool the reviewer *should* have is still there, so the exclusion is
    # targeted rather than a blanket removal of the GENERIC group.
    assert "read_file" in names
    assert "get_financials" in names


def test_missing_key_surfaces_as_a_structured_error(tmp_path):
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
    )
    adapter = TavilyAdapter(api_key=None)
    access = DataAccess([adapter], cache=LocalCache(tmp_path / "cache"), settings=settings)
    tool = WebSearchTool(access)

    result = run(tool.run(query="茅台"))

    assert result.ok is False
    assert "未配置" in result.error


def test_search_result_is_cached_so_a_repeat_does_not_refetch(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return search_handler(request)

    access = make_access(tmp_path, handler)

    async def both():
        first = await access.web_search("茅台", top_n=2)
        second = await access.web_search("茅台", top_n=2)
        return first, second

    first, second = run(both())

    assert calls["n"] == 1
    assert second.from_cache is True
    assert first.from_cache is False


def test_the_two_operations_use_separate_cache_slots(tmp_path):
    """Search and fetch share kind='web'; the op field must keep them apart."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/extract"):
            return httpx.Response(
                200, json={"results": [{"url": "https://x.com/a", "raw_content": "正文"}]}
            )
        return search_handler(request)

    access = make_access(tmp_path, handler)

    async def both():
        await access.web_search("茅台", top_n=2)
        return await access.fetch_url("https://x.com/a")

    fetched = run(both())

    assert fetched.from_cache is False
    assert fetched.endpoint.endswith("extract")
