"""Two-tier registry: resident/lazy split, search and activation (docs 03.4.3)."""

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
    # The schema appears only after activation, matching "the model may only
    # call tools it has been given".
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

    # A name hit is worth more than a description hit, so a tool whose *name*
    # carries the keyword scores at least the name-match weight.
    briefs = {brief.name: brief.score for brief in registry.search("quote", limit=10)}
    assert briefs.get("get_quote", 0) >= 3


def test_search_matches_chinese_keywords_in_descriptions(tmp_path):
    registry = make_registry(tmp_path)

    # Chinese keywords live in the descriptions; both valuation-related tools
    # should surface for a 估值 query.
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
        "calc_metrics", "read_file", "write_file", "research_plan",
        "search_tools", "list_skills", "load_skill", "load_tool", "ask_user",
    }
    assert set(registry.names()) == expected
    assert set(registry.names()) == {cls.name for cls in ALL_TOOL_CLASSES}


def test_meta_tools_can_reach_the_catalogue(tmp_path):
    """search_tools and load_tool operate on the registry they live in."""
    registry = make_registry(tmp_path)
    tool = registry.resolve("search_tools")

    assert tool.registry is registry
