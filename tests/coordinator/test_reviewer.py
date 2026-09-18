"""risk-review sub-agent 的测试（文档 03.10）。"""

from __future__ import annotations

import asyncio

import pandas as pd

from finharness.config.settings import ContextSettings, Settings
from finharness.coordinator import Coordinator
from finharness.coordinator.reviewer import REVIEW_MAX_TURNS
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.cost import SessionStats
from finharness.provider.base import Provider
from finharness.tools.registry import review_tool_names
from finharness.types import (
    ModelUsage,
    StreamChunk,
    StreamEvent,
    ToolUse,
)


class Adapter(DataAdapter):
    """最小的 fetch 接口：足以让 reviewer 重新获取一次 quote。"""

    name = "fake"

    def fetch_quote(self, symbol):
        return FetchResult(
            df=pd.DataFrame([{"symbol": symbol, "close": 100.0}]),
            interface="fake_quote",
        )


class ScriptedProvider(Provider):
    """回放预设轮次，并记录 sub-agent 发出的每次请求。"""

    def __init__(self, rounds=None, *, error: Exception | None = None):
        self.rounds = list(rounds or [])
        self.error = error
        self.requests: list[dict] = []

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
        self.requests.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": [entry["function"]["name"] for entry in tools],
            }
        )
        if self.error is not None:
            raise self.error
        script = self.rounds.pop(0) if self.rounds else [message_end()]
        for chunk in script:
            yield chunk


def message_end(*tool_uses: ToolUse) -> StreamChunk:
    return StreamChunk(
        StreamEvent.MESSAGE_END,
        ModelUsage(input_tokens=5, output_tokens=2, tool_uses=list(tool_uses)),
    )


def text_round(*parts: str) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in parts] + [message_end()]


def tool_round(*tool_uses: ToolUse) -> list[StreamChunk]:
    return [message_end(*tool_uses)]


def make_settings(tmp_path, **context) -> Settings:
    return Settings(
        context=ContextSettings(**context),
        data={"cache_dir": tmp_path / "cache"},
        paths={"output_dir": tmp_path / "output"},
    )


def make_coordinator(tmp_path, provider, *, settings=None, stats=None, usage=None, cite=None):
    settings = settings or make_settings(tmp_path)
    data = DataAccess(
        [Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings
    )
    coordinator = Coordinator(
        provider=provider,
        data=data,
        settings=settings,
        cite=cite if cite is not None else CitationRegistry(),
    )
    if stats is not None or usage is not None:
        coordinator.bind_accounting(
            usage=usage if usage is not None else ModelUsage(),
            stats=stats if stats is not None else SessionStats(),
        )
    return coordinator


def run(coro):
    return asyncio.run(coro)


# -- 受限的 tool 目录 ---------------------------------------------------------


def test_review_catalogue_is_read_only_data_tools_plus_read_file():
    names = set(review_tool_names())

    assert {"get_quote", "get_financials", "get_indicators", "read_file"} <= names
    # Writes、output tools 以及所有 META tool 都被排除：它们都会让 reviewer
    # 扩大自己的目录或改动会话状态。
    assert not {"write_report", "write_file", "make_chart"} & names
    assert not {
        "research_plan",
        "remember_preference",
        "search_tools",
        "ask_user",
    } & names
    # 纯计算器无法获取任何数据，因此不增加校验能力。
    assert not {"calc_metrics", "calc_valuation"} & names


# -- 隔离 ---------------------------------------------------------------------


def test_review_reads_the_checklist_and_cannot_see_the_main_transcript(tmp_path):
    provider = ScriptedProvider([text_round("未发现实质性问题")])
    coordinator = make_coordinator(tmp_path, provider)

    result = run(coordinator.review_risk(topic="贵州茅台", markdown="# 贵州茅台\n\n正文"))

    assert result.ok is True
    request = provider.requests[0]
    assert "风险终审" in request["system"]
    assert "风险核查清单" in request["system"]
    # 报告是 reviewer 的唯一输入：恰好一条 user 消息。
    assert len(request["messages"]) == 1
    assert request["messages"][0].role == "user"
    assert "贵州茅台" in request["messages"][0].content


def test_review_request_carries_the_report_body(tmp_path):
    provider = ScriptedProvider([text_round("ok")])
    coordinator = make_coordinator(tmp_path, provider)

    run(coordinator.review_risk(topic="t", markdown="## 风险提示\n\n1. 偿债压力"))

    assert "偿债压力" in provider.requests[0]["messages"][0].content


def test_review_uses_only_the_restricted_tool_schemas(tmp_path):
    provider = ScriptedProvider([text_round("ok")])
    coordinator = make_coordinator(tmp_path, provider)

    run(coordinator.review_risk(topic="t", markdown="body"))

    assert set(provider.requests[0]["tools"]) == set(review_tool_names())


def test_review_leaves_no_conversation_memory_behind(tmp_path):
    """sub-agent 没有存储，因此不会污染对话作用域。"""
    provider = ScriptedProvider([text_round("ok")])
    coordinator = make_coordinator(tmp_path, provider)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    # review 是一次片段，而非记忆：除了它自己的 summary 什么都不会返回。
    assert result.summary == "ok"


# -- 子代理审计（docs 03.7.3）--------------------------------------------------


def test_subagent_tool_calls_are_audited_with_user_id(tmp_path):
    """子代理不再是审计盲区：它的工具调用以 focus 前缀的 session 落行，
    并带上与主会话相同的 user_id。"""
    import json

    from finharness.hooks.audit import AuditHook, AuditLogWriter

    provider = ScriptedProvider(
        [
            tool_round(ToolUse("c1", "get_quote", {"symbol": "600519"})),
            text_round("未发现实质性问题"),
        ]
    )
    settings = make_settings(tmp_path)
    data = DataAccess([Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)
    audit_path = tmp_path / "audit.jsonl"
    writer = AuditLogWriter(audit_path)
    coordinator = Coordinator(
        provider=provider,
        data=data,
        settings=settings,
        cite=CitationRegistry(),
        audit_hook_factory=lambda session_id: AuditHook(
            writer, session_id=session_id, user_id="u_test"
        ),
        user_id="u_test",
    )

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert result.ok is True
    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_rows = [row for row in rows if row.get("action") == "run"]
    assert tool_rows, "子代理的工具调用必须留下审计行"
    assert all(row["user_id"] == "u_test" for row in tool_rows)
    assert all(row["session_id"].endswith("-subagent") for row in tool_rows)
    assert tool_rows[0]["tool"] == "get_quote"


def test_subagent_runs_without_an_audit_factory(tmp_path):
    """未接线审计工厂时（测试替身路径）子代理照常运行，只是不留痕。"""
    provider = ScriptedProvider([text_round("ok")])
    coordinator = make_coordinator(tmp_path, provider)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert result.ok is True


# -- 预算 ---------------------------------------------------------------------


def test_reviewer_turn_budget_is_tightened_without_touching_the_main_settings(tmp_path):
    settings = make_settings(tmp_path, max_turns=30)
    # 不同的调用次数多于 reviewer 的预算：不同参数可避开 loop guard，
    # 因此唯一能终止运行的就是 reviewer 自身的 turn 预算。
    provider = ScriptedProvider(
        [
            *[tool_round(ToolUse(f"c{i}", "get_quote", {"symbol": f"60000{i}"})) for i in range(8)],
            text_round("done"),
        ]
    )
    coordinator = make_coordinator(tmp_path, provider, settings=settings)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert len(provider.requests) == REVIEW_MAX_TURNS
    assert result.turns == REVIEW_MAX_TURNS
    # 主会话的预算未受影响。
    assert settings.context.max_turns == 30


def test_reviewer_budget_leaves_room_to_verify_several_figures(tmp_path):
    """读取加上若干独立核对必须放得下，否则 review 会空着发布。"""
    assert REVIEW_MAX_TURNS >= 5


# -- 账目统计 -----------------------------------------------------------------


def test_review_folds_its_tokens_into_the_session_totals_and_breakdown(tmp_path):
    provider = ScriptedProvider([text_round("意见"), text_round("再一轮")])
    stats = SessionStats()
    usage = ModelUsage()
    coordinator = make_coordinator(tmp_path, provider, stats=stats, usage=usage)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert usage.input_tokens == 5
    assert stats.input_tokens == 5
    assert stats.snapshot().per_agent["risk"] == {
        "input_tokens": 5,
        "output_tokens": 2,
        "runs": 1,
    }
    # reviewer 的 tool 调用不得出现在主 loop 的 per-tool 视图中。
    assert "get_quote" not in stats.snapshot().per_tool


def test_review_citations_continue_the_session_numbering(tmp_path):
    settings = make_settings(tmp_path)
    cite = CitationRegistry()
    # 一个已存在的 citation，仿佛主 agent 已经获取过数据。
    cite.register(
        tool="get_quote",
        endpoint="fake:fake_quote",
        symbol="600519",
        params={},
        rows=1,
        cols=2,
        fingerprint="fp",
    )
    provider = ScriptedProvider(
        [tool_round(ToolUse("c1", "get_quote", {"symbol": "600000"})), text_round("意见")]
    )
    coordinator = make_coordinator(tmp_path, provider, settings=settings, cite=cite)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert result.citations == ["cit_000002"]
    assert cite.get("cit_000002") is not None


def test_review_without_accounting_wiring_still_runs(tmp_path):
    provider = ScriptedProvider([text_round("意见")])
    coordinator = make_coordinator(tmp_path, provider)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert result.ok is True
    assert result.input_tokens == 5


# -- 失败隔离 -----------------------------------------------------------------


def test_provider_failure_degrades_to_a_structured_error(tmp_path):
    provider = ScriptedProvider(error=RuntimeError("boom"))
    coordinator = make_coordinator(tmp_path, provider)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert result.ok is False
    assert result.error is not None
    assert "boom" in result.error
    assert result.summary == ""


def test_failed_review_charges_no_tokens_to_the_session(tmp_path):
    """没有花费任何东西的运行不得虚增总量。

    失败的尝试仍会被 *计入* 为一次运行——这是关于 review 失败频率的有用信号——
    但它不得移动任何 token。
    """
    provider = ScriptedProvider(error=RuntimeError("boom"))
    stats = SessionStats()
    usage = ModelUsage()
    coordinator = make_coordinator(tmp_path, provider, stats=stats, usage=usage)

    run(coordinator.review_risk(topic="t", markdown="body"))

    snapshot = stats.snapshot()
    assert snapshot.input_tokens == 0
    assert snapshot.output_tokens == 0
    assert usage.input_tokens == 0
    assert snapshot.per_agent["risk"]["runs"] == 1


def test_subagent_tokens_are_tagged_once_and_not_double_counted(tmp_path):
    """子代理的模型调用以 call_type=subagent 记一次；协调器只汇总量、不重发指标。

    否则同一次调用会在 llm_tokens_total 里出现两次，成本视图直接翻倍。
    """
    from finharness.observability.metrics import MetricsRecorder
    from finharness.observability.observer import Observer

    metrics = MetricsRecorder()
    observer = Observer(metrics=metrics)
    provider = ScriptedProvider([text_round("意见"), text_round("再一轮")])
    stats = SessionStats()
    usage = ModelUsage()
    settings = make_settings(tmp_path)
    data = DataAccess([Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)
    coordinator = Coordinator(
        provider=provider, data=data, settings=settings, cite=CitationRegistry()
    )
    coordinator.bind_accounting(usage=usage, stats=stats, observer=observer)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert result.input_tokens == 5
    # 会话总量只累加一次：token 指标由子循环自己发出，协调器只汇总量，不重发。
    assert stats.input_tokens == 5
    rendered = metrics.render().decode()
    assert 'llm_tokens_total{call_type="subagent",kind="input"' in rendered
    assert rendered.count('llm_tokens_total{call_type="subagent",kind="input",model="unknown"} 5.0') == 1
    # 子代理不是用户请求：请求级直方图不得因此出现样本。
    assert "agent_request_duration_seconds_count" not in rendered
