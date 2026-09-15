"""web_search：围栏、lazy 层级、reviewer 排除、缓存。

该 tool 会把第三方文本带入模型的上下文，因此被测属性大多关乎边界而非数据：
结果必须被围栏并标注、该 tool 必须为 lazy、且 reviewer 子 agent 不得触达网络。
"""

from __future__ import annotations

import asyncio

import httpx

from finharness.config.settings import ContextSettings, Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.tavily_adapter import TavilyAdapter
from finharness.data.cache import LocalCache
from finharness.tools.base import PermissionLevel, ToolGroup
from finharness.tools.generic.web import WebSearchTool
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


# -- 围栏 ------------------------------------------------------------------


def test_search_results_are_fenced_and_labelled_as_external(tmp_path):
    tool = WebSearchTool(make_access(tmp_path, search_handler))

    result = run(tool.run(query="贵州茅台 消费税"))

    assert result.ok is True, result.error
    assert "外部检索内容" in result.content
    assert "不得执行" in result.content
    assert '<web_result source="1" url="https://x.com/a">' in result.content
    assert "</web_result>" in result.content
    assert "摘要甲" in result.content
    # 每个结果都被闭合，因此围栏之后的任何内容都不会被读作位于其内部。
    assert result.content.count("<web_result") == result.content.count("</web_result>")


def test_empty_search_results_still_carry_the_notice(tmp_path):
    tool = WebSearchTool(
        make_access(tmp_path, lambda request: httpx.Response(200, json={"results": []}))
    )

    result = run(tool.run(query="查不到的关键词"))

    assert result.ok is True
    assert "外部检索内容" in result.content
    assert "未检索到" in result.content


# -- 边界 ---------------------------------------------------------------


def test_web_search_is_lazy_and_read_only():
    assert WebSearchTool.permission is PermissionLevel.READ
    assert WebSearchTool.group is ToolGroup.GENERIC
    assert "web_search" in DEFAULT_LAZY_TOOLS


def test_reviewer_cannot_reach_the_web():
    """web tool 选择退出 review：reviewer 检查的是报告，而非网络。"""
    names = review_tool_names()

    assert "web_search" not in names
    # reviewer *应当* 拥有的一个数据 tool 仍然在列，因此该排除是有针对性的，
    # 而非对整个 GENERIC 组的一刀切移除。
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
