"""端到端验收：自然语言 -> agent 选择工具 -> 有依据的回复。

这些测试会访问真实的 LLM provider 和真实的市场数据源，因此被标记为
``smoke``，并且在缺少凭证和网络时会被跳过。
运行方式：``uv run pytest -m smoke``；默认测试集不包含它们。
"""

from __future__ import annotations

import asyncio
import os

import pytest

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
# 使用随包发布的 prompt，以便该资产本身也处于测试之下。
SYSTEM_PROMPT = system_prompt()


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def _build_loop(tmp_path) -> AgentLoop:
    import httpx

    provider = OpenAICompatProvider(
        base_url=DEEPSEEK_BASE_URL,
        api_key=_api_key(),
        model=DEEPSEEK_MODEL,
        client=httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0)),
    )
    data = DataAccess(
        [AkShareAdapter(throttle_seconds=0.2)],
        cache=LocalCache(tmp_path / "cache"),
        settings=Settings(),
    )
    return AgentLoop(
        provider=provider,
        registry=ToolRegistry(data),
        settings=Settings(),
        system=SYSTEM_PROMPT,
        cite=CitationRegistry(),
    )


def test_natural_language_question_routes_to_a_tool_and_returns_grounded_data(tmp_path):
    """一个价格问题必须触达市场数据工具，并返回一个真实数字。"""
    loop = _build_loop(tmp_path)

    async def run():
        return await asyncio.wait_for(loop.run("贵州茅台(600519)最新股价是多少？只回答价格。"), 180)

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    assert outcome.tool_calls >= 1, "the model should have called a data tool"
    # 有依据的回答包含从获取的数据帧中得到的数字。
    assert any(ch.isdigit() for ch in outcome.answer)
    # 已为实际使用的数据记录来源信息。
    assert outcome.citations, "a data-backed answer must register citations"


def test_natural_language_kline_question_uses_history_tool(tmp_path):
    loop = _build_loop(tmp_path)

    async def run():
        return await asyncio.wait_for(loop.run("帮我看看贵州茅台(600519)近一年的股价走势。"), 180)

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    assert outcome.tool_calls >= 1
    assert outcome.citations
