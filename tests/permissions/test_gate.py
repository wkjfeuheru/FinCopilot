"""Permission gate 行为：模式、deny 规则、路径白名单（文档 03.7.1）。"""

import asyncio
from dataclasses import dataclass

from finharness.config.settings import PermissionSettings, Settings
from finharness.permissions.gate import PermissionGate, ReadOnlyGate
from finharness.permissions.modes import Verdict
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
    """非交互调用方必须拒绝，而不是静默放行。"""
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
    """即使在 read tool 上，夹带的交易指令也会被捕获。"""
    decision = check(FakeTool(), {"keyword": "帮我下单 买入"}, make_settings(tmp_path))
    assert decision.verdict is Verdict.DENY
    assert "条规则" in decision.reason


def test_deny_rule_reports_which_rule_matched(tmp_path):
    decision = check(FakeTool(), {"code": "import easytrader"}, make_settings(tmp_path))
    assert decision.verdict is Verdict.DENY
    assert "easytrader" in decision.reason


# -- 嵌套扫描与只读闸门（docs 03.7.2）----------------------------------------

def test_deny_rule_reaches_nested_structures(tmp_path):
    """交易意图藏在 list[dict] 内层也要被捕获：只扫顶层是可绕过的。"""
    args = {"sections": [{"title": "概览"}, {"body": "建议立即下单"}]}

    decision = check(FakeTool(), args, make_settings(tmp_path))

    assert decision.verdict is Verdict.DENY


def test_deny_rule_reaches_nested_dict_keys(tmp_path):
    args = {"payload": {"买入(600519)": "x"}}

    decision = check(FakeTool(), args, make_settings(tmp_path))

    assert decision.verdict is Verdict.DENY


def test_scanning_is_depth_bounded(tmp_path):
    """过深的结构不再扫描，但不得抛异常（成本有界）。"""
    deep: dict = {"leaf": "帮我下单"}
    for _ in range(40):
        deep = {"nested": deep}

    decision = check(FakeTool(), deep, make_settings(tmp_path))

    assert decision.verdict is Verdict.ALLOW


def test_custom_deny_patterns_override_the_builtin_set(tmp_path):
    """``deny_patterns`` 不再是死参数：传入后按传入集判定。"""
    gate = PermissionGate(
        settings=make_settings(tmp_path), deny_patterns=(r"内幕消息",)
    )

    banned = asyncio.run(gate.check(FakeTool(), {"q": "给我内幕消息"}))
    # 内置集被替换，因此内置命中不再生效。
    builtin_only = asyncio.run(gate.check(FakeTool(), {"q": "帮我下单"}))

    assert banned.verdict is Verdict.DENY
    assert builtin_only.verdict is Verdict.ALLOW


def test_readonly_gate_still_applies_deny_rules(tmp_path):
    """只读闸门描述"不改状态"，不描述"入参可信"。"""
    gate = ReadOnlyGate()

    decision = asyncio.run(gate.check(FakeTool(), {"keyword": "帮我下单 买入"}))

    assert decision.verdict is Verdict.DENY
    assert "条规则" in decision.reason


def test_readonly_gate_allows_clean_read_tools():
    decision = asyncio.run(ReadOnlyGate().check(FakeTool(), {"symbol": "600519"}))

    assert decision.verdict is Verdict.ALLOW


# -- 缓存拒写与网络外发确认（docs 03.7.1）--------------------------------------


def test_write_into_cache_dir_is_denied(tmp_path):
    """缓存子树对读取开放、对写入关闭：lookup 键跨用户共享，写入即投毒。"""
    settings = make_settings(tmp_path)
    target = str(settings.data.cache_dir / "parquet" / "2401-01" / "abc.parquet")

    for mode in ("default", "auto"):
        decision = check(
            FakeTool(permission=PermissionLevel.WRITE),
            {"path": target},
            make_settings(tmp_path, mode=mode),
        )
        assert decision.verdict is Verdict.DENY
        assert "缓存" in decision.reason


def test_egress_tools_require_confirmation(tmp_path):
    """web_search 是读类工具，但网络外发首次须经确认。"""

    tool = FakeTool(name="web_search")
    decision = check(tool, {"query": "白酒政策"}, make_settings(tmp_path))
    assert decision.verdict is Verdict.DENY
    assert "网络" in decision.reason


def test_egress_allows_after_user_confirms(tmp_path):
    tool = FakeTool(name="web_search")

    async def confirm(name, args):
        return True

    decision = check(tool, {"query": "白酒政策"}, make_settings(tmp_path), confirm)
    assert decision.verdict is Verdict.ALLOW
    assert "已确认网络" in decision.reason


def test_egress_remembered_within_conversation(tmp_path):
    """确认回调记录类别后，本对话内后续外发免问。"""

    confirmed: set[str] = set()

    async def confirm_egress(name, args):
        confirmed.add("egress")  # 模拟"允许并本对话不再询问"
        return True

    gate = PermissionGate(
        settings=make_settings(tmp_path),
        confirm=confirm_egress,
        conversation_id="conv-1",
        confirmed_categories=confirmed,
        confirm_egress=confirm_egress,
    )

    first = asyncio.run(gate.check(FakeTool(name="web_search"), {"query": "a"}))
    second = asyncio.run(gate.check(FakeTool(name="web_search"), {"query": "b"}))

    assert first.verdict is Verdict.ALLOW
    assert second.verdict is Verdict.ALLOW
    assert "本对话已确认" in second.reason


def test_egress_denial_is_not_remembered(tmp_path):
    """用户拒绝后不记录类别：下次外发仍要问。"""

    confirmed: set[str] = set()

    async def confirm_egress(name, args):
        return False

    gate = PermissionGate(
        settings=make_settings(tmp_path),
        confirm_egress=confirm_egress,
        conversation_id="conv-1",
        confirmed_categories=confirmed,
    )

    first = asyncio.run(gate.check(FakeTool(name="web_search"), {"query": "a"}))

    assert first.verdict is Verdict.DENY
    assert confirmed == set()  # 拒绝不进免问集合


def test_egress_allows_in_auto_mode(tmp_path):
    class WebTool(FakeTool):
        name = "web_search"

    decision = check(WebTool(), {"query": "a"}, make_settings(tmp_path, mode="auto"))
    assert decision.verdict is Verdict.ALLOW


def test_research_reports_metadata_mode_still_direct(tmp_path):
    """研报工具不带 with_text 时仍是普通读调用，直通。"""

    tool = FakeTool(name="get_research_reports")
    decision = check(
        tool, {"industry": "证券Ⅱ"}, make_settings(tmp_path)
    )
    assert decision.verdict is Verdict.ALLOW
    assert decision.reason == ""


def test_research_reports_full_text_requires_confirmation(tmp_path):
    tool = FakeTool(name="get_research_reports")
    decision = check(
        tool, {"with_text": True}, make_settings(tmp_path)
    )
    assert decision.verdict is Verdict.DENY
    assert "网络" in decision.reason
