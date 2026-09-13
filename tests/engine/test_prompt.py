"""The system prompt is a product asset; guard what it must contain.

It governs when the agent plans, how it cites, and when it stops. If a future
edit drops one of those sections the agent silently loses that behaviour, so the
required directives are asserted here.
"""

import pytest

from finharness.engine.prompt import PROMPT_PATH, PromptNotFoundError, system_prompt


@pytest.fixture(autouse=True)
def _clear_cache():
    system_prompt.cache_clear()
    yield
    system_prompt.cache_clear()


def test_prompt_asset_exists_and_is_substantial():
    text = system_prompt()

    assert len(text) > 300
    assert PROMPT_PATH.is_file()


def test_prompt_defines_when_to_plan():
    text = system_prompt()

    assert "research_plan" in text
    # It must say when NOT to plan, or trivial questions pay for a plan turn.
    assert "不要" in text or "无需" in text
    assert "简单事实" in text


def test_prompt_carries_the_citation_and_traceability_rules():
    text = system_prompt()

    assert "{cite:" in text          # the placeholder syntax
    assert "citation" in text.lower()
    assert "溯源" in text or "回溯" in text
    assert "无来源" in text           # the unsourced-number marker


def test_prompt_requires_convergence():
    """Convergence is the direct antidote to the over-running sessions."""
    text = system_prompt()

    assert "收敛" in text
    assert "重复" in text


def test_prompt_routes_skills_and_reports():
    text = system_prompt()

    assert "load_skill" in text
    assert "report-template" in text
    assert "write_report" in text


def test_prompt_states_when_not_to_produce_deliverables():
    """Without this boundary the agent upgrades an analysis into a report itself."""
    text = system_prompt()

    assert "调用边界" in text
    assert "make_chart" in text
    # It must say the deliverable is opt-in, not the default outcome of analysis.
    assert "明确要求" in text
    assert "不要" in text


def test_prompt_flags_an_unreviewed_report():
    text = system_prompt()

    assert "未经独立复核" in text


def test_prompt_states_output_discipline():
    text = system_prompt()

    assert "买卖建议" in text


def test_missing_asset_raises(tmp_path):
    with pytest.raises(PromptNotFoundError):
        system_prompt(tmp_path / "nope.md")


def test_empty_asset_raises(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")

    with pytest.raises(PromptNotFoundError):
        system_prompt(empty)


def test_server_uses_the_same_prompt_as_the_loader():
    """One definition: the shipped prompt and the server constant must agree."""
    from finharness.server.api import DEFAULT_SYSTEM_PROMPT

    assert DEFAULT_SYSTEM_PROMPT == system_prompt()
