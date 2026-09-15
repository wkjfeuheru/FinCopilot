"""两级 registry：resident/lazy 拆分、搜索与激活（docs 03.4.3）。"""

from finharness.config.settings import Settings, ToolSettings
from finharness.data.access import DataAccess
from finharness.tools.registry import ALL_TOOL_CLASSES, ToolRegistry


def make_registry(tmp_path, **tool_kwargs) -> ToolRegistry:
    settings = Settings(tools=ToolSettings(**tool_kwargs), data={"cache_dir": tmp_path / "cache"})
    return ToolRegistry(DataAccess([], settings=settings), settings=settings)


def schema_names(registry: ToolRegistry) -> list[str]:
    return [entry["function"]["name"] for entry in registry.schemas()]


def test_lazy_tools_are_not_injected_until_activated(tmp_path):
    registry = make_registry(tmp_path)

    assert "get_announcements" in registry.lazy_names()
    assert "get_announcements" not in schema_names(registry)

    assert registry.activate("get_announcements") is True
    # schema 只有在激活之后才出现，这与"模型只能调用
    # 已授予它的 tool"一致。
    assert "get_announcements" in schema_names(registry)


def test_activating_twice_is_idempotent(tmp_path):
    registry = make_registry(tmp_path)
    registry.activate("get_announcements")

    assert registry.activate("get_announcements") is False


def test_activating_an_unknown_tool_is_refused(tmp_path):
    registry = make_registry(tmp_path)

    assert registry.activate("no_such_tool") is False


def test_explicit_resident_list_overrides_the_default(tmp_path):
    registry = make_registry(tmp_path, resident=("get_quote", "get_kline"))

    assert set(registry.resident_names()) == {"get_quote", "get_kline"}
    assert "get_indicators" in registry.lazy_names()
    assert schema_names(registry) == ["get_quote", "get_kline"]


def test_search_matches_name_and_description(tmp_path):
    registry = make_registry(tmp_path)

    names = [brief.name for brief in registry.search("公告")]
    assert "get_announcements" in names


def test_search_ranks_a_name_match_above_a_description_match(tmp_path):
    registry = make_registry(tmp_path)

    # 命中名称比命中 description 更有价值，因此 *name* 中带有该关键字的 tool
    # 至少获得名称匹配的权重。
    briefs = {brief.name: brief.score for brief in registry.search("quote", limit=10)}
    assert briefs.get("get_quote", 0) >= 3


def test_search_matches_chinese_keywords_in_descriptions(tmp_path):
    registry = make_registry(tmp_path)

    # 中文关键字存在于 description 中；两个与 valuation 相关的 tool
    # 都应在 估值 查询中出现。
    scores = {brief.name: brief.score for brief in registry.search("估值", limit=10)}
    assert scores.get("get_valuation", 0) >= 1
    assert scores.get("get_peers", 0) >= 1


def test_search_returns_nothing_for_an_unrelated_query(tmp_path):
    registry = make_registry(tmp_path)

    assert registry.search("zzzzz") == []


def test_every_tool_is_either_resident_or_lazy(tmp_path):
    registry = make_registry(tmp_path)
    assert set(registry.resident_names()) | set(registry.lazy_names()) == set(registry.names())


def test_registry_covers_the_documented_catalogue(tmp_path):
    registry = make_registry(tmp_path)
    expected = {
        "get_quote", "get_kline", "get_indicators", "get_financials",
        "get_valuation", "get_peers", "get_market_news", "get_announcements",
        "get_research_reports",
        "get_macro_indicators", "get_industry_perf", "get_industry_constituents",
        "calc_metrics", "calc_valuation", "run_backtest", "make_chart", "write_report",
        "read_file", "write_file", "web_search", "research_plan",
        "update_plan_step", "record_conclusion",
        "search_tools", "list_skills", "load_skill", "load_tool", "spawn_agent",
        "ask_user", "remember_preference",
    }
    assert set(registry.names()) == expected
    assert set(registry.names()) == {cls.name for cls in ALL_TOOL_CLASSES}


def test_the_general_sub_agent_gets_local_material_tools_only(tmp_path):
    """worker 消费素材；它不得携带数据获取层。

    reviewer 合理地拥有数据 tool（它要复核数字）；普通 worker 则不然，
    因为任务语句不指定任何 symbol 或 period，若赋予它数据 tool，
    会招致猜测而非隔离。
    """
    from finharness.tools.registry import review_tool_names, worker_tool_names

    worker = set(worker_tool_names())
    reviewer = set(review_tool_names())

    assert worker == {"read_file"}
    assert "get_quote" not in worker and "get_research_reports" not in worker
    # reviewer 保留其数据 tool——这正是它的全部价值所在。
    assert "get_quote" in reviewer
    # 且两个子 agent 都不得再 spawn 或触达 META/写 tool。
    for name in ("spawn_agent", "write_report", "write_file", "load_tool", "ask_user"):
        assert name not in worker
        assert name not in reviewer


def test_meta_tools_can_reach_the_catalogue(tmp_path):
    """search_tools 与 load_tool 操作它们自身所在的 registry。"""
    registry = make_registry(tmp_path)
    tool = registry.resolve("search_tools")

    assert tool.registry is registry
