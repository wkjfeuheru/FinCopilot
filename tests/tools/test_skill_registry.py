"""Skill registry semantics: semantic contract, idempotence, observability,
and merged tool+skill discovery (docs 03.8 + user principles)."""

import asyncio

from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.tools.meta.discovery import SearchToolsTool
from finharness.tools.meta.skills import LoadSkillTool, SkillRegistry


def packaged() -> SkillRegistry:
    return SkillRegistry(Settings().paths.skills_dir)


def test_catalogue_has_the_documented_eight_skills():
    names = set(packaged().names())
    assert names == {
        "dupont-analysis", "dcf-valuation", "valuation-comps", "earnings-quality",
        "risk-checklist", "industry-framework", "backtest-protocol", "report-template",
    }


def test_every_skill_declares_the_semantic_contract():
    """Each skill documents what it does, its contract, when to use it, and a sample."""
    for meta in packaged().list_skills():
        assert meta.description, meta.name
        assert meta.inputs, f"{meta.name} 缺少 inputs"
        assert meta.outputs, f"{meta.name} 缺少 outputs"
        assert meta.use_cases, f"{meta.name} 缺少 use_cases"
        assert meta.examples, f"{meta.name} 缺少 examples"
        assert meta.allowed_tools, f"{meta.name} 缺少 allowed_tools"


def test_related_skills_all_resolve_to_real_skills():
    """A composed-skill reference must not dangle."""
    registry = packaged()
    known = set(registry.names())
    for meta in registry.list_skills():
        for related in meta.related_skills:
            assert related in known, f"{meta.name} 引用了不存在的技能：{related}"


def test_report_template_composes_the_analysis_skills():
    template = packaged().get("report-template")
    # The aggregator must point at the analysis skills it organises.
    assert {"dupont-analysis", "dcf-valuation", "risk-checklist"} <= set(template.related_skills)


def test_load_is_idempotent_and_reports_reuse():
    registry = packaged()

    _, _, first = registry.load("dupont-analysis")
    _, _, second = registry.load("dupont-analysis")

    assert first.reused is False
    assert second.reused is True
    assert registry.is_loaded("dupont-analysis")


def test_load_history_is_observable():
    """Observability: every load leaves a timestamped, timed record."""
    registry = packaged()
    registry.load("dcf-valuation")
    registry.load("dcf-valuation")

    history = registry.load_history()

    assert len(history) == 2
    assert all(record.name == "dcf-valuation" for record in history)
    assert all(record.ts for record in history)
    assert all(record.duration_ms >= 0 for record in history)
    assert [record.reused for record in history] == [False, True]


def test_identical_loads_produce_identical_bodies():
    """Idempotence is what makes a retried tool call safe."""
    registry = packaged()
    _, first, _ = registry.load("risk-checklist")
    _, second, _ = registry.load("risk-checklist")

    assert first == second


def test_search_matches_use_cases_and_examples_not_just_description():
    registry = packaged()

    results = dict((meta.name, score) for meta, score in registry.search("研报"))
    assert "report-template" in results


def test_search_ranks_a_name_match_highest():
    registry = packaged()

    results = registry.search("dupont")
    assert results and results[0][0].name == "dupont-analysis"


def test_dependency_hint_names_composed_skills_without_loading_them():
    registry = packaged()
    template = registry.get("report-template")

    hint = registry.dependency_hint(template)

    assert "dupont-analysis" in hint
    assert "load_skill" in hint


def test_load_skill_tool_surfaces_contract_and_dependency_hint():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    data = DataAccess([])
    data.settings = Settings()

    result = asyncio.run(LoadSkillTool(data, ctx=ctx).run(name="report-template"))

    assert result.ok is True
    assert "产出：" in result.content
    assert "本技能可复用" in result.content
    assert ctx.loaded_skills == ["report-template"]


def test_load_skill_tool_reports_reuse_on_second_call():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    data = DataAccess([])
    data.settings = Settings()

    async def run():
        tool = LoadSkillTool(data, ctx=ctx)
        await tool.run(name="dupont-analysis")
        return await tool.run(name="dupont-analysis")

    result = asyncio.run(run())

    assert "复用" in result.content


def test_search_tools_returns_both_kinds_in_one_result_set():
    """One question, one result set: tools and skills are found together."""
    data = DataAccess([])
    data.settings = Settings()
    registry = _registry_with_tools()

    result = asyncio.run(SearchToolsTool(data, registry=registry).run(query="估值"))

    assert "[工具]" in result.content
    assert "[技能]" in result.content


def _registry_with_tools():
    from finharness.tools.registry import ToolRegistry

    settings = Settings()
    return ToolRegistry(DataAccess([], settings=settings), settings=settings)
