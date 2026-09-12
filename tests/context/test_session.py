"""ResearchContext: plan lifecycle, conclusions, and system injection (docs 03.6.2)."""

import pytest

from finharness.config.settings import Settings
from finharness.context.session import PlanStep, ResearchContext
from finharness.data.citation import CitationRegistry


def make_ctx() -> ResearchContext:
    return ResearchContext(cite=CitationRegistry(), settings=Settings())


def test_symbols_follow_the_citation_registry():
    ctx = make_ctx()
    ctx.cite.register(
        tool="get_quote", endpoint="e", symbol="600519", params={}, rows=1, cols=1, fingerprint="x"
    )

    assert ctx.symbols == ["600519"]


def test_set_plan_installs_then_revises():
    ctx = make_ctx()

    first = ctx.set_plan("分析茅台", [PlanStep(seq=1, action="取行情")])
    assert first.revision == 1

    second = ctx.set_plan("分析茅台并对比", [PlanStep(seq=1, action="取行情")])
    assert second.revision == 2
    assert second.plan_id == first.plan_id  # same plan, new revision


def test_mark_plan_step_updates_status():
    ctx = make_ctx()
    ctx.set_plan("g", [PlanStep(seq=1, action="a"), PlanStep(seq=2, action="b")])

    assert ctx.mark_plan_step(1, "done") is True
    assert ctx.plan.steps[0].status == "done"
    assert ctx.mark_plan_step(99, "done") is False


def test_invalid_step_status_is_rejected():
    ctx = make_ctx()
    ctx.set_plan("g", [PlanStep(seq=1, action="a")])

    with pytest.raises(ValueError):
        ctx.mark_plan_step(1, "unknown")


def test_plan_digest_renders_status_marks():
    ctx = make_ctx()
    ctx.set_plan("分析茅台", [PlanStep(seq=1, action="取行情", tool_hint=["get_quote"])])
    ctx.mark_plan_step(1, "done")

    digest = ctx.plan_digest()
    assert "✓" in digest
    assert "get_quote" in digest


def test_plan_digest_is_empty_without_a_plan():
    assert make_ctx().plan_digest() == ""


def test_add_skill_is_idempotent():
    ctx = make_ctx()

    assert ctx.add_skill("dupont-analysis") is True
    assert ctx.add_skill("dupont-analysis") is False
    assert ctx.loaded_skills == ["dupont-analysis"]


def test_activate_tool_is_idempotent():
    ctx = make_ctx()

    assert ctx.activate_tool("get_announcements") is True
    assert ctx.activate_tool("get_announcements") is False


def test_conclusions_keep_their_cids():
    ctx = make_ctx()

    conclusion = ctx.add_conclusion("ROE 约 30%", ["cit_000001"])

    assert conclusion.cids == ["cit_000001"]
    assert ctx.conclusions[0].text == "ROE 约 30%"
    assert conclusion.ts  # timestamped


def test_state_block_is_empty_when_there_is_nothing_to_report():
    assert make_ctx().state_block() == ""


def test_state_block_includes_plan_skills_and_conclusions():
    ctx = make_ctx()
    ctx.set_plan("分析茅台", [PlanStep(seq=1, action="取行情")])
    ctx.add_skill("dupont-analysis")
    ctx.add_conclusion("ROE 高", ["cit_000001"])

    block = ctx.state_block()

    assert "研究计划" in block
    assert "dupont-analysis" in block
    assert "cit_000001" in block
