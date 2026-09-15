"""M2 的端到端验收：planning、meta 工具与 governance。

会访问真实 provider 和真实市场数据，因此标记为 ``smoke``。
运行方式：``uv run pytest -m smoke``。
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from finharness.config.settings import PermissionSettings, Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import HookChain
from finharness.permissions.gate import PermissionGate
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.smoke

# 使用随包发布的 prompt，以便该资产本身也处于测试之下。
SYSTEM_PROMPT = system_prompt()


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return key


def _build(tmp_path, *, planning_system: str = SYSTEM_PROMPT):
    import httpx

    settings = Settings(
        permission=PermissionSettings(default_mode="auto"),
        data={"cache_dir": tmp_path / "cache"},
        audit={"log_path": tmp_path / "audit.jsonl"},
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
    writer = AuditLogWriter(settings.audit.log_path)
    audit = AuditHook(writer, session_id="e2e")
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry(data, ctx=ctx, settings=settings),
        settings=settings,
        system=planning_system,
        cite=citations,
        ctx=ctx,
        gate=PermissionGate(settings=settings),
        hooks=HookChain([audit]),
    )
    loop.audit = audit
    return loop, ctx, writer


def test_complex_question_produces_a_plan_and_grounded_conclusion(tmp_path):
    """多步问题应经由 research_plan 和真实数据来回答。"""
    loop, ctx, _ = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(
            loop.run("对比贵州茅台(600519)近一年的盈利趋势，说明 ROE 变化的原因。"), 300
        )

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    # 对于复杂问题，plan 是 M2 的验收锚点。
    assert ctx.plan is not None, "a multi-step question should install a plan"
    assert ctx.plan.steps, "the plan should contain steps"
    assert outcome.tool_calls >= 1, "the model should have fetched data"
    assert outcome.citations, "conclusions must be traceable to citations"


def test_simple_question_answers_without_planning(tmp_path):
    """简单的查询不应为 research_plan 花费一个轮次。"""
    loop, ctx, _ = _build(tmp_path)

    async def run():
        return await asyncio.wait_for(loop.run("600519 最新股价是多少？"), 240)

    outcome = asyncio.run(run())

    assert outcome.succeeded is True, outcome.error
    assert outcome.tool_calls >= 1, "should still fetch the quote"
    # 事实查询不需要 planning（见文档 03.6.2 的路由）。
    assert ctx.plan is None


def test_audit_log_records_the_whole_session(tmp_path):
    loop, _, writer = _build(tmp_path)

    async def run():
        loop.audit.session_start(mode="auto", provider="OpenAICompatProvider", model="deepseek-chat")
        outcome = await asyncio.wait_for(loop.run("600519 最新股价是多少？"), 240)
        snapshot = loop.stats.snapshot()
        loop.audit.session_end(
            total_tokens=snapshot.input_tokens + snapshot.output_tokens,
            tool_calls=snapshot.tool_calls,
        )
        return outcome

    outcome = asyncio.run(run())
    records = [
        json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines() if line
    ]
    actions = [r["action"] for r in records]

    assert outcome.succeeded is True, outcome.error
    assert actions[0] == "session_start"
    assert actions[-1] == "session_end"
    assert "run" in actions, "tool executions must be audited"
    run_record = next(r for r in records if r["action"] == "run")
    assert run_record["tool"]
    assert run_record["verdict"] == "allow"
