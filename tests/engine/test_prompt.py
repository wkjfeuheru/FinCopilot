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


# --- role, capability boundary and refusal policy (prompt design pass) --------

def test_prompt_defines_a_specific_role_and_capability_boundary():
    """A vague "you are an assistant" leaves the model free to improvise."""
    text = system_prompt()

    assert "角色与能力边界" in text
    assert "A 股" in text
    # Both sides must be stated, or the model cannot tell what to decline.
    assert "你能做" in text or "能做" in text
    assert "不能做" in text or "不能" in text
    # The named gaps are what makes the boundary operational.
    assert "港股" in text or "美股" in text
    assert "预测" in text


def test_prompt_states_the_knowledge_boundary():
    """Params carry data; the model's own memory must not be a number source."""
    text = system_prompt()

    assert "知识边界" in text
    assert "本次取数" in text


def test_prompt_shows_the_citation_format_by_example():
    """A bare description is weaker than description + a concrete example."""
    text = system_prompt()

    assert "正确示例" in text
    assert "错误示例" in text
    # The examples must actually use the placeholder syntax they teach.
    assert text.count("{cite:") >= 3


def test_prompt_report_example_matches_the_real_schema():
    """The JSON example must keep validating against ReportInput.

    An example that drifts from the schema teaches the model a shape the tool
    will reject, which is worse than having no example at all.
    """
    import json
    import re

    from finharness.tools.fin.writer import ReportInput

    blocks = re.findall(r"```json\n(.*?)```", system_prompt(), re.DOTALL)
    assert blocks, "the prompt should carry a JSON example for write_report"

    for block in blocks:
        ReportInput.model_validate(json.loads(block))


def test_prompt_has_a_refusal_policy():
    """Without this the model answers out of range instead of declining.

    Declining is a consequence of the capability boundary, so the policy lives
    inside that section rather than as a topic of its own; the test asserts that
    placement, not just the wording's presence somewhere in the file.
    """
    text = system_prompt()
    role_section = text.split("## 角色与能力边界", 1)[1].split("\n## ", 1)[0]

    assert "做不到" in role_section
    assert "不要勉强" in role_section or "不硬撑" in role_section
    assert "ask_user" in role_section          # missing input -> clarify
    # The anti-hallucination rule, stated as the bottom line.
    assert "不得猜测" in role_section or "不得编造" in role_section
    assert "取不到" in role_section


def test_prompt_has_no_standalone_refusal_section():
    """Guards the consolidation: the policy must not drift back out on its own."""
    text = system_prompt()

    assert "## 超出能力范围时" not in text


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
