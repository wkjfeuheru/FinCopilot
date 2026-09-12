"""Permission gate behaviour: modes, deny rules, path whitelist (docs 03.7.1)."""

import asyncio
from dataclasses import dataclass

import pytest

from finharness.config.settings import PermissionSettings, Settings
from finharness.permissions.gate import PermissionGate, ReadOnlyGate
from finharness.permissions.modes import PermissionMode, Verdict
from finharness.tools.base import PermissionLevel


@dataclass
class FakeTool:
    name: str = "t"
    permission: PermissionLevel = PermissionLevel.READ


def make_settings(tmp_path, *, mode="default") -> Settings:
    return Settings(
        permission=PermissionSettings(default_mode=mode),
        data={"cache_dir": tmp_path / "cache"},
    )


def check(tool, args, settings=None, confirm=None):
    gate = PermissionGate(settings=settings, confirm=confirm)
    return asyncio.run(gate.check(tool, args))


def test_read_tools_are_allowed_in_every_mode(tmp_path):
    for mode in ("default", "plan", "auto"):
        decision = check(FakeTool(), {}, make_settings(tmp_path, mode=mode))
        assert decision.verdict is Verdict.ALLOW


def test_default_readonly_gate_denies_write_tools():
    gate = ReadOnlyGate()
    decision = asyncio.run(gate.check(FakeTool(permission=PermissionLevel.WRITE), {}))
    assert decision.verdict is Verdict.DENY
    assert "not read-only" in decision.reason


def test_write_tool_confirms_under_default_mode(tmp_path):
    async def confirm(name, args):
        return True

    decision = check(
        FakeTool(permission=PermissionLevel.WRITE), {}, make_settings(tmp_path), confirm
    )
    assert decision.verdict is Verdict.ALLOW
    assert "确认" in decision.reason


def test_write_tool_denied_when_user_refuses(tmp_path):
    async def confirm(name, args):
        return False

    decision = check(
        FakeTool(permission=PermissionLevel.WRITE), {}, make_settings(tmp_path), confirm
    )
    assert decision.verdict is Verdict.DENY


def test_write_tool_denied_without_an_interactive_channel(tmp_path):
    """Non-interactive callers must refuse rather than silently allow."""
    decision = check(FakeTool(permission=PermissionLevel.WRITE), {}, make_settings(tmp_path))
    assert decision.verdict is Verdict.DENY
    assert "无交互通道" in decision.reason


def test_auto_mode_allows_write_tools(tmp_path):
    decision = check(
        FakeTool(permission=PermissionLevel.WRITE), {}, make_settings(tmp_path, mode="auto")
    )
    assert decision.verdict is Verdict.ALLOW


def test_write_into_output_dir_is_whitelisted(tmp_path):
    settings = make_settings(tmp_path)
    target = str(settings.paths.output_dir / "report.md")
    decision = check(
        FakeTool(permission=PermissionLevel.WRITE), {"path": target}, settings
    )
    assert decision.verdict is Verdict.ALLOW
    assert "白名单" in decision.reason


def test_path_traversal_out_of_the_whitelist_is_not_allowed(tmp_path):
    settings = make_settings(tmp_path)
    escaped = str(settings.paths.output_dir / ".." / ".." / "secrets.txt")
    decision = check(
        FakeTool(permission=PermissionLevel.WRITE), {"path": escaped}, settings
    )
    assert decision.verdict is Verdict.DENY


def test_deny_rule_blocks_trading_intent_in_arguments(tmp_path):
    """A smuggled trade instruction is caught even on a read tool."""
    decision = check(FakeTool(), {"keyword": "帮我下单 买入"}, make_settings(tmp_path))
    assert decision.verdict is Verdict.DENY
    assert "条规则" in decision.reason


def test_deny_rule_reports_which_rule_matched(tmp_path):
    decision = check(FakeTool(), {"code": "import easytrader"}, make_settings(tmp_path))
    assert decision.verdict is Verdict.DENY
    assert "easytrader" in decision.reason
