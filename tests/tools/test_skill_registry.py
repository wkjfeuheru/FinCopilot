"""场景 skill 目录：语义契约、随附文件枚举、按目标幂等加载、可观测性，
以及工具与 skill 的合并检索。"""

import asyncio

from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.tools.meta.discovery import SearchToolsTool
from finharness.tools.meta.skills import LoadSkillTool, SkillError, SkillRegistry

SCENARIOS = ("equity-research", "industry-research", "macro-research", "quant-factor")


def packaged() -> SkillRegistry:
    return SkillRegistry(Settings().paths.skills_dir)


def test_catalogue_has_the_four_scenario_skills():
    names = set(packaged().names())
    assert names == set(SCENARIOS)


def test_risk_checklist_is_a_prompt_asset_not_a_skill():
    """reviewer 的检查标准已迁移至 prompts/；它不得作为 skill 再次出现。"""
    assert "risk-checklist" not in packaged().names()
    from finharness.engine.prompt import risk_checklist_prompt

    assert "风险" in risk_checklist_prompt()


def test_every_scenario_declares_the_semantic_contract():
    """每个 skill 都记录其用途、契约、适用场景以及示例。"""
    for meta in packaged().list_skills():
        assert meta.description, meta.name
        assert meta.inputs, f"{meta.name} 缺少 inputs"
        assert meta.outputs, f"{meta.name} 缺少 outputs"
        assert meta.use_cases, f"{meta.name} 缺少 use_cases"
        assert meta.examples, f"{meta.name} 缺少 examples"
        assert meta.allowed_tools, f"{meta.name} 缺少 allowed_tools"
        assert meta.content_estimate > 0, meta.name


def test_every_scenario_ships_references_and_a_report_template():
    for meta in packaged().list_skills():
        assert any(f.startswith("references/") for f in meta.files), meta.name
        assert "assets/report-template.md" in meta.files, meta.name


def test_enumerated_files_exist_on_disk():
    """目录是唯一事实来源：不存在声明了却缺失的文件。"""
    from pathlib import Path

    registry = packaged()
    for meta in registry.list_skills():
        skill_dir = Path(meta.path).parent
        for rel in meta.files:
            assert (skill_dir / rel).is_file(), f"{meta.name} 声明的文件缺失：{rel}"


def test_load_skill_flow_is_idempotent_and_reports_reuse():
    registry = packaged()

    _, _, first = registry.load("equity-research")
    _, _, second = registry.load("equity-research")

    assert first.reused is False
    assert second.reused is True
    assert registry.is_loaded("equity-research")


def test_load_reference_file_returns_its_body():
    registry = packaged()

    meta, body, record = registry.load("equity-research", file="references/valuation.md")

    assert meta.name == "equity-research"
    assert record.name == "equity-research/references/valuation.md"
    assert "DCF" in body
    assert record.reused is False


def test_loading_a_file_is_tracked_independently_of_the_flow():
    """flow 与 file 是不同的加载目标，各自独立去重。"""
    registry = packaged()

    registry.load("equity-research")
    _, _, file_record = registry.load("equity-research", file="references/valuation.md")

    assert file_record.reused is False  # 流程加载不得掩盖文件加载
    _, _, second_file = registry.load("equity-research", file="references/valuation.md")
    assert second_file.reused is True


def test_unknown_file_is_refused_with_the_available_list():
    registry = packaged()

    try:
        registry.load("equity-research", file="references/nope.md")
    except SkillError as exc:
        assert "references/valuation.md" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown file should raise SkillError")


def test_load_history_is_observable_across_targets():
    registry = packaged()
    registry.load("quant-factor")
    registry.load("quant-factor", file="references/single-series.md")

    history = registry.load_history()

    assert [record.name for record in history] == [
        "quant-factor",
        "quant-factor/references/single-series.md",
    ]
    assert all(record.ts for record in history)
    assert all(record.duration_ms >= 0 for record in history)


def test_identical_loads_produce_identical_bodies():
    registry = packaged()
    _, first, _ = registry.load("industry-research", file="references/prosperity.md")
    _, second, _ = registry.load("industry-research", file="references/prosperity.md")

    assert first == second


def test_search_matches_scenarios_by_use_cases_and_examples():
    registry = packaged()

    results = {meta.name for meta, _score in registry.search("估值")}
    assert "equity-research" in results


def test_search_ranks_a_name_match_highest():
    registry = packaged()

    results = registry.search("macro")
    assert results and results[0][0].name == "macro-research"


def test_describe_lists_loadable_files():
    text = packaged().describe()

    assert "equity-research" in text
    assert "references/valuation.md" in text
    assert "assets/report-template.md" in text


def test_load_skill_tool_surfaces_contract_and_records_flow():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    data = DataAccess([])
    data.settings = Settings()

    result = asyncio.run(LoadSkillTool(data, ctx=ctx).run(name="equity-research"))

    assert result.ok is True
    assert "产出：" in result.content
    assert ctx.loaded_skills == ["equity-research"]


def test_load_skill_tool_loads_a_file_and_records_the_target():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    data = DataAccess([])
    data.settings = Settings()

    result = asyncio.run(
        LoadSkillTool(data, ctx=ctx).run(name="equity-research", file="references/valuation.md")
    )

    assert result.ok is True
    assert "参考文件：equity-research/references/valuation.md" in result.content
    assert ctx.loaded_skills == ["equity-research/references/valuation.md"]


def test_load_skill_tool_reports_reuse_on_second_call():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    data = DataAccess([])
    data.settings = Settings()

    async def run():
        tool = LoadSkillTool(data, ctx=ctx)
        await tool.run(name="quant-factor")
        return await tool.run(name="quant-factor")

    result = asyncio.run(run())

    assert "复用" in result.content


def test_search_tools_returns_both_kinds_in_one_result_set():
    """一次提问，一个结果集：工具与 skill 一并被检索到。"""
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
