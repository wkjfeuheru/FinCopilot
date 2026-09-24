"""通过 AgentLoop 触达的治理链：gate、hooks、audit、ctx 注入。"""

import asyncio
import json
import sys
from pathlib import Path

# 复用 engine 的测试替身；测试套件没有 tests 包，因此通过路径加载
# 兄弟模块。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_loop import (  # noqa: E402
    RecordingTool,
    ScriptedProvider,
    StubRegistry,
    text_round,
    tool_round,
)

from finharness.config.settings import (
    ContextSettings,
    PermissionSettings,
    Settings,
    ToolSettings,
)
from finharness.context.session import ResearchContext
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import BaseHook, HookChain
from finharness.permissions.gate import PermissionGate
from finharness.tools.base import PermissionLevel
from finharness.types import ToolUse


def make_loop(provider, *, registry, settings=None, gate=None, hooks=None, ctx=None, output=None):
    return AgentLoop(
        provider=provider,
        registry=registry,
        settings=settings or Settings(),
        system="sys",
        output=output,
        gate=gate,
        hooks=hooks,
        ctx=ctx,
    )


def test_gate_denial_blocks_execution_and_reports_a_reason():
    async def run():
        tool = RecordingTool("danger", content="should not run")
        registry = StubRegistry({"danger": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("c1", "danger", {})), text_round("ok")]
        )

        # 一个始终拒绝的 gate，与工具权限无关。
        class DenyAll:
            async def check(self, tool, args):
                from finharness.permissions.gate import GateDecision
                from finharness.permissions.modes import Verdict

                return GateDecision(Verdict.DENY, "forbidden by policy")

        loop = make_loop(provider, registry=registry, gate=DenyAll())
        outcome = await loop.run("do it")
        return outcome, loop.messages, tool

    outcome, messages, tool = asyncio.run(run())

    assert tool.calls == []  # 从未执行
    payload = json.loads(messages[2].tool_results[0][1])
    assert payload["ok"] is False
    assert "forbidden by policy" in payload["error"]
    assert outcome.succeeded is True  # 模型仍然得以作答


def test_pre_hook_can_block_a_tool():
    class BlockingHook(BaseHook):
        async def pre(self, tool, args, *, turn=0):
            return False

    async def run():
        tool = RecordingTool("t", content="x")
        registry = StubRegistry({"t": tool})
        provider = ScriptedProvider([tool_round(ToolUse("c1", "t", {})), text_round("ok")])
        loop = make_loop(provider, registry=registry, hooks=HookChain([BlockingHook()]))
        await loop.run("q")
        return tool

    assert asyncio.run(run()).calls == []


def test_audit_hook_records_run_and_session_boundaries(tmp_path):
    async def run():
        writer = AuditLogWriter(tmp_path / "audit.jsonl")
        audit = AuditHook(writer, session_id="s_x")
        audit.session_start(mode="default", provider="ScriptedProvider", model="m")
        tool = RecordingTool("get_quote", content="报价")
        registry = StubRegistry({"get_quote": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("c1", "get_quote", {"symbol": "600519"})), text_round("ok")]
        )
        loop = make_loop(provider, registry=registry, hooks=HookChain([audit]))
        await loop.run("茅台报价")
        audit.session_end(total_tokens=9, tool_calls=1)
        return tmp_path / "audit.jsonl"

    path = asyncio.run(run())
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [r["action"] for r in records] == ["session_start", "run", "session_end"]
    run_record = records[1]
    assert run_record["tool"] == "get_quote"
    assert run_record["verdict"] == "allow"
    assert "600519" in run_record["args_summary"]
    assert run_record["duration_ms"] >= 0


def test_write_tool_confirmation_flows_through_the_gate(tmp_path):
    """写类工具会到达 confirm 回调，若被拒绝则不会执行。"""
    seen = []

    async def confirm(name, args):
        seen.append(name)
        return False

    settings = Settings(
        permission=PermissionSettings(default_mode="default"),
        paths={"output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )

    async def run():
        tool = RecordingTool("write_note", permission=PermissionLevel.WRITE, content="wrote")
        registry = StubRegistry({"write_note": tool})
        provider = ScriptedProvider(
            [tool_round(ToolUse("c1", "write_note", {"path": "x"})), text_round("ok")]
        )
        gate = PermissionGate(settings=settings, confirm=confirm)
        loop = make_loop(provider, registry=registry, settings=settings, gate=gate)
        outcome = await loop.run("写一条")
        return outcome, loop.messages, tool

    outcome, messages, tool = asyncio.run(run())

    assert seen == ["write_note"]
    assert tool.calls == []
    payload = json.loads(messages[2].tool_results[0][1])
    assert payload["ok"] is False
    assert "拒绝" in payload["error"]


def test_context_state_is_injected_as_the_trailing_request_message():
    """状态随请求附在历史之后，而不是放在 system prompt 中。

    将其排除在 ``system`` 之外，才能让 provider 的前缀缓存同时保留
    静态 prompt *和* 不断增长的历史；若把它放在最前面，会把缓存边界
    移到对话的开头（docs 3.3）。
    """
    from finharness.context.session import PlanStep

    async def run():
        ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
        ctx.set_plan("分析茅台", [PlanStep(seq=1, action="取行情")])
        ctx.add_skill("equity-research")
        provider = ScriptedProvider([text_round("ok")])
        loop = make_loop(provider, registry=StubRegistry(), ctx=ctx)
        await loop.run("问题")
        return provider

    provider = asyncio.run(run())

    request = provider.requests[0]
    # system prompt 保持静态……
    assert "研究计划" not in request["system"]
    assert "equity-research" not in request["system"]
    # ……而状态改为作为最后一条消息到达。
    assert request["roles"][-1] == "user"
    state = request["messages"][-1].content
    assert "研究计划" in state
    assert "equity-research" in state


def test_tool_timeout_uses_the_declared_value_when_settings_has_no_override():
    """工具自身的 timeout 必须被遵守（settings 的覆盖仍然优先）。"""
    from finharness.types import AgentTurnOutcome  # noqa: F401 - 仅为清晰起见而导入

    async def run():
        slow = RecordingTool("slow", delay=5.0)
        slow.timeout = 1  # 声明的预算：1s
        registry = StubRegistry({"slow": slow})
        provider = ScriptedProvider([tool_round(ToolUse("c1", "slow", {})), text_round("ok")])
        settings = Settings(
            tools=ToolSettings(timeout_default_s=30),  # 全局默认值不会触发 5s 超时
            context=ContextSettings(max_turns=3),
        )
        loop = make_loop(provider, registry=registry, settings=settings)
        outcome = await loop.run("慢工具")
        return outcome, loop.messages

    outcome, messages = asyncio.run(run())

    payload = json.loads(messages[2].tool_results[0][1])
    assert payload["ok"] is False
    assert "timeout" in payload["error"]
    assert "1s" in payload["error"]  # 声明的是 1s，而非 30s 默认值
