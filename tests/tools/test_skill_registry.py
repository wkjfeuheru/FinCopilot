"""场景 skill 目录：语义契约、随附文件枚举、按目标幂等加载、可观测性，
场景路由，以及工具与 skill 的合并检索。"""

import asyncio

from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.shared.capabilities import Capability
from finharness.tools.meta.discovery import SearchToolsTool
from finharness.tools.meta.skills import SkillError, SkillRegistry, route

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


def test_loading_the_scenario_flow_is_idempotent_and_reports_reuse():
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


def test_a_single_capability_earns_methodology_but_not_the_scenario_flow():
    """简单提问的答案就在这里。

    问"ROE 为什么掉这么多"只命中财务能力：它拿到盈利能力方法论，但不被塞进一份完整的
    权益研究流程——流程描述的是多步骤编排，对单点提问是无用开销。
    """
    targets = route(capabilities={Capability.FINANCIAL})

    keys = [target.key for target in targets]
    assert "equity-research/references/profitability.md" in keys
    assert "equity-research" not in keys


def test_crossing_the_capability_threshold_also_injects_the_scenario_flow():
    """需要多项能力时，编排本身才是答案的一部分。"""
    targets = route(capabilities={Capability.FINANCIAL, Capability.VALUATION})

    keys = [target.key for target in targets]
    assert "equity-research" in keys
    assert "equity-research/references/profitability.md" in keys
    assert "equity-research/references/valuation.md" in keys


def test_a_plan_skill_hint_is_honoured_over_inference():
    """计划显式点名了就照给——模型对场景的判断优先于关键词推断。"""
    targets = route(capabilities={Capability.MARKET}, hinted=("macro-research",))

    keys = [target.key for target in targets]
    assert "macro-research" in keys


def test_route_is_empty_when_nothing_is_recognised():
    assert route(capabilities=set()) == []


def test_every_routed_target_resolves_to_a_real_file():
    """路由表是手写的，因此它必须与磁盘上的技能包对得上。"""
    from pathlib import Path

    registry = packaged()
    cases = [
        {Capability.FINANCIAL},
        {Capability.VALUATION},
        {Capability.PEER},
        {Capability.INDUSTRY},
        {Capability.MACRO},
        {Capability.COMPUTE},
        {Capability.FINANCIAL, Capability.VALUATION},
    ]
    for capabilities in cases:
        for target in route(capabilities=capabilities):
            meta = registry.get(target.skill)
            assert meta is not None, f"路由指向不存在的技能：{target.skill}"
            if target.file:
                skill_dir = Path(meta.path).parent
                assert (skill_dir / target.file).is_file(), target.key


def test_injection_into_the_context_is_idempotent():
    """同一目标只注入一次，因此一条追问不会重复占用预算。"""
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())

    assert ctx.inject_methodology("equity-research", "正文") is True
    assert ctx.inject_methodology("equity-research", "正文") is False
    assert ctx.methodology == {"equity-research": "正文"}
    assert ctx.loaded_skills == ["equity-research"]


def test_injected_methodology_is_rendered_into_the_state_block():
    """注入的正文必须真的进入模型可见的状态块，否则等于没注入。"""
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    ctx.inject_methodology("equity-research/references/valuation.md", "DCF 要看自由现金流")

    block = ctx.state_block()

    assert "方法论：equity-research/references/valuation.md" in block
    assert "DCF 要看自由现金流" in block


def test_search_tools_returns_both_kinds_in_one_result_set():
    """一次提问，一个结果集：工具与 skill 一并被检索到。"""
    data = DataAccess([])
    data.settings = Settings()
    registry = _registry_with_tools()

    result = asyncio.run(SearchToolsTool(data, registry=registry).run(query="估值"))

    assert "[工具]" in result.content
    assert "[技能]" in result.content


def test_search_activates_the_lazy_tools_it_returns():
    """检索即激活：命中之后，下一次请求里它的 schema 已经就位。"""
    data = DataAccess([])
    data.settings = Settings()
    registry = _registry_with_tools()
    assert registry.is_active("get_announcements") is False

    result = asyncio.run(SearchToolsTool(data, registry=registry).run(query="公告"))

    assert registry.is_active("get_announcements") is True
    assert "已激活" in result.content


def test_search_result_carries_the_parameter_list():
    """参数清单让模型一次检索就能发起调用，而不必等下一轮的完整 schema。"""
    data = DataAccess([])
    data.settings = Settings()
    registry = _registry_with_tools()

    result = asyncio.run(SearchToolsTool(data, registry=registry).run(query="公告"))

    assert "参数：" in result.content
    assert "symbol" in result.content


def test_a_direct_call_to_a_lazy_tool_is_no_longer_a_rejection():
    """按需注入是省 token 的手段，不是权限：闸门改为就地激活并放行。

    这里只断言注册层的行为——循环的执行路径由 engine 侧测试覆盖。
    """
    registry = _registry_with_tools()

    assert registry.is_lazy("get_announcements") is True
    assert registry.is_active("get_announcements") is False

    assert registry.activate("get_announcements") is True
    assert registry.is_active("get_announcements") is True
    # 重复激活不改变状态。
    assert registry.activate("get_announcements") is False


def test_a_scoped_registry_has_no_lazy_tier():
    """子代理拿到的是一个固定工作集，按需激活在那里只会让目录随运行漂移。"""
    from finharness.tools.registry import ToolRegistry

    settings = Settings()
    scoped = ToolRegistry(
        DataAccess([], settings=settings),
        settings=settings,
        only={"get_quote", "read_file"},
    )

    assert scoped.lazy_names() == []
    assert set(scoped.resident_names()) == {"get_quote", "read_file"}


def _registry_with_tools():
    from finharness.tools.registry import ToolRegistry

    settings = Settings()
    return ToolRegistry(DataAccess([], settings=settings), settings=settings)
