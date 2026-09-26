"""tool_status 的可展示摘要：白名单投影，禁止原样倾倒 args。"""

from finharness.engine.tool_summary import tool_status_summary


def test_industry_and_symbol_are_copied():
    assert tool_status_summary(
        "get_industry_perf", {"industry": "白酒", "years": 1}
    ) == {"industry": "白酒", "years": 1}
    assert tool_status_summary("get_quote", {"symbol": "600519"}) == {"symbol": "600519"}


def test_unknown_keys_and_file_bodies_are_dropped():
    summary = tool_status_summary(
        "write_file",
        {
            "path": "/tmp/reports/研报.md",
            "content": "SECRET_BODY" * 40,
            "api_key": "sk-live",
        },
    )
    assert summary == {"file": "研报.md"}
    assert "SECRET_BODY" not in str(summary)
    assert "api_key" not in summary


def test_empty_or_unknown_args_yield_empty_summary():
    assert tool_status_summary("experimental_private_tool", {"foo": "bar"}) == {}
    assert tool_status_summary("get_quote", None) == {}
    assert tool_status_summary("get_quote", {}) == {}


def test_spawn_tasks_are_first_lines_truncated():
    long_task = "分析 600519 盈利质量\n第二行不应出现"
    summary = tool_status_summary(
        "spawn_agent",
        {"tasks": [long_task, "分析 000858 估值"], "context": "共享背景请勿下发"},
    )
    assert summary["tasks"][0] == "分析 600519 盈利质量"
    assert summary["tasks"][1] == "分析 000858 估值"
    assert "context" not in summary
    assert "第二行" not in summary["tasks"][0]
