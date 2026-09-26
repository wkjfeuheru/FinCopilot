"""M4 验收：在长时间的真实会话中，window 保持有界。

WBS M4 的要求是「10 轮而不撑爆 window」。因此这里要检验的主张是关于 *window* 的，
而不是开放式研究任务能否在轮次预算内完成——那取决于模型是否重新获取数据，
属于 L2 的职责（已推迟）。因此这里断言：

* 一旦记录增长，compaction 就会启动，
* 它把 window 拉回预算之内，且
* 最终的 window 仍处于模型的限制之内。
"""

from __future__ import annotations

import asyncio
import os

import pytest

from finharness.config.settings import ContextSettings, PermissionSettings, Settings
from finharness.context.session import ResearchContext
from finharness.context.tokens import TokenCounter
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.permissions.gate import PermissionGate
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke

# 使用随包发布的 prompt，以便该资产本身也处于测试之下。
SYSTEM_PROMPT = system_prompt()

# 故意设得较小，使常规的多步任务会越过它并必须触发 compact——但不能小到连
# "压不动的地板"都装不下。窗口的固定开销是 system prompt + 全部工具 schema
# （实测约 12.5k token），压缩只删历史消息、删不掉这部分。若阈值
# （compaction_ratio × 窗口）落到地板之下，压缩在数学上无法达标，断言
# "_window_tokens() < WINDOW_TOKENS" 就恒为假——那验的是配置写错，而不是
# "窗口有界"这一契约。32000 让阈值（19200）明显高于地板，留有真实余量。
WINDOW_TOKENS = 32000
COMPACTION_RATIO = 0.6


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def _build(tmp_path):
    import httpx

    settings = Settings(
        permission=PermissionSettings(default_mode="auto"),
        context=ContextSettings(
            context_window_tokens=WINDOW_TOKENS,
            compaction_ratio=COMPACTION_RATIO,
            max_turns=30,
        ),
        data={"cache_dir": tmp_path / "cache"},
        paths={"output_dir": tmp_path / "output"},
    )
    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1",
        api_key=_api_key(),
        model="deepseek-chat",
        client=httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0)),
    )
    data = DataAccess(
        [AkShareAdapter(throttle_seconds=0.2)], cache=LocalCache(tmp_path / "cache"), settings=settings
    )
    citations = CitationRegistry()
    ctx = ResearchContext(cite=citations, settings=settings)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=SYSTEM_PROMPT,
        cite=citations,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        counter=TokenCounter(cache_dir=tmp_path / "cache" / "tiktoken"),
    )
    return loop, settings


def test_long_session_keeps_the_window_bounded(tmp_path):
    """数据密集的会话必须触发 compact，并保持在 window 之内。"""
    loop, _ = _build(tmp_path)

    # 先自检窗口配置本身是自洽的：阈值必须高于不可压缩地板，否则下面的断言
    # 无论如何都不可能成立，失败原因会指向压缩逻辑而非"窗口设小了"。
    assert loop.memory.threshold_is_reachable(
        system=SYSTEM_PROMPT, tools=loop.registry.schemas()
    ), (
        f"窗口配置自相矛盾：固定地板 "
        f"{loop.memory.fixed_tokens(system=SYSTEM_PROMPT, tools=loop.registry.schemas())} token "
        f"已 ≥ 压缩阈值 {loop.memory.compaction_threshold()}；上调 WINDOW_TOKENS"
    )

    async def run():
        return await asyncio.wait_for(
            loop.run(
                "请研究贵州茅台(600519)：先看最新行情与近一年股价走势，"
                "再看财务指标与杜邦分解，再看同业估值对比。"
            ),
            600,
        )

    outcome = asyncio.run(run())

    # 临时的 provider 故障意味着 window 从未被真正触发；
    # 应如实报告，而不是空泛地让测试通过。
    if outcome.reason == "provider_error":
        pytest.skip(f"provider error, window not exercised: {outcome.error}")

    # 该次运行可能在轮次预算内完成，也可能没有，但它绝不能
    # 撑爆 window——这正是 M4 所保证的。
    hard_limit = WINDOW_TOKENS
    assert loop.compactions, "the session should have grown enough to compact"
    assert loop._window_tokens() < hard_limit, (
        f"window {loop._window_tokens()} exceeded the {hard_limit} limit"
    )
    for result in loop.compactions:
        assert result.after_tokens < hard_limit, "compaction must leave room to continue"
        assert result.compacted is True
    # 要么它已收敛，要么它在保持 window 较小的前提下用尽了轮次。
    assert outcome.succeeded is True or outcome.reason == "max_turns_exhausted"


def test_compaction_engages_on_a_multi_step_task(tmp_path):
    """越过小 window 的任务必须真正触发 compaction。"""
    loop, _ = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(
            loop.run(
                "对比 600519 与 000858 的行情、财务指标与估值，并给出风险提示。"
            ),
            600,
        )

    asyncio.run(run())

    assert loop.compactions, "a multi-step data task should exceed the small window"
    first = loop.compactions[0]
    # 这种缩减必须是实质性的，而非偶然。
    assert first.after_tokens < first.before_tokens
    assert first.removed > 0
