"""Tests for the risk-review sub-agent (docs 03.10)."""

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
    """Minimal fetch surface: enough for the reviewer to re-fetch a quote."""

    name = "fake"

    def fetch_quote(self, symbol):
        return FetchResult(
            df=pd.DataFrame([{"symbol": symbol, "close": 100.0}]),
            interface="fake_quote",
        )


class ScriptedProvider(Provider):
    """Replays canned rounds and records every request the sub-agent makes."""

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


# -- the restricted catalogue -------------------------------------------------


def test_review_catalogue_is_read_only_data_tools_plus_read_file():
    names = set(review_tool_names())

    assert {"get_quote", "get_financials", "get_indicators", "read_file"} <= names
    # Writes, output tools and every META tool are excluded. load_tool matters
    # most: it would let the reviewer widen its own catalogue.
    assert not {"write_report", "write_file", "make_chart"} & names
    assert not {
        "research_plan",
        "remember_preference",
        "load_tool",
        "search_tools",
        "ask_user",
        "load_skill",
        "list_skills",
    } & names
    # Pure calculators cannot fetch anything, so they add no verification power.
    assert not {"calc_metrics", "calc_valuation"} & names


# -- isolation ----------------------------------------------------------------


def test_review_reads_the_checklist_and_cannot_see_the_main_transcript(tmp_path):
    provider = ScriptedProvider([text_round("未发现实质性问题")])
    coordinator = make_coordinator(tmp_path, provider)

    result = run(coordinator.review_risk(topic="贵州茅台", markdown="# 贵州茅台\n\n正文"))

    assert result.ok is True
    request = provider.requests[0]
    assert "风险终审" in request["system"]
    assert "风险核查清单" in request["system"]
    # The report is the reviewer's only input: exactly one user message.
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
    """The sub-agent has no store, so it cannot pollute the conversation scope."""
    provider = ScriptedProvider([text_round("ok")])
    coordinator = make_coordinator(tmp_path, provider)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    # A review is an episode, not memory: nothing but its own summary comes back.
    assert result.summary == "ok"


# -- budget -------------------------------------------------------------------


def test_reviewer_turn_budget_is_tightened_without_touching_the_main_settings(tmp_path):
    settings = make_settings(tmp_path, max_turns=30)
    # More distinct calls than the reviewer's budget: distinct arguments keep the
    # loop guard out of the way, so the only thing that can stop the run is the
    # reviewer's own turn budget.
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
    # The main session's budget is untouched.
    assert settings.context.max_turns == 30


def test_reviewer_budget_leaves_room_to_verify_several_figures(tmp_path):
    """The read plus several independent checks must fit, or reviews ship empty."""
    assert REVIEW_MAX_TURNS >= 5


# -- accounting ---------------------------------------------------------------


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
    # The reviewer's tool calls must not appear in the main loop's per-tool view.
    assert "get_quote" not in stats.snapshot().per_tool


def test_review_citations_continue_the_session_numbering(tmp_path):
    settings = make_settings(tmp_path)
    cite = CitationRegistry()
    # A pre-existing citation, as if the main agent had already fetched data.
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


# -- failure isolation --------------------------------------------------------


def test_provider_failure_degrades_to_a_structured_error(tmp_path):
    provider = ScriptedProvider(error=RuntimeError("boom"))
    coordinator = make_coordinator(tmp_path, provider)

    result = run(coordinator.review_risk(topic="t", markdown="body"))

    assert result.ok is False
    assert result.error is not None
    assert "boom" in result.error
    assert result.summary == ""


def test_failed_review_charges_no_tokens_to_the_session(tmp_path):
    """A run that spent nothing must not inflate the totals.

    The failed attempt is still *counted* as a run — that is useful signal about
    how often review fails — but it must move no tokens.
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
