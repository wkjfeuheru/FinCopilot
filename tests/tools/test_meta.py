"""Skill catalogue, meta tools and the interactive bus."""

import asyncio

import pytest

from finharness.config.settings import Settings
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.server.confirm import ConfirmBus
from finharness.tools.meta.skills import (
    ListSkillsTool,
    LoadSkillTool,
    SkillError,
    SkillLibrary,
)


def packaged_skills() -> SkillLibrary:
    return SkillLibrary(Settings().paths.skills_dir)


def test_packaged_catalogue_lists_both_sample_skills():
    names = {meta.name for meta in packaged_skills().list_skills()}
    assert {"dupont-analysis", "report-template"} <= names


def test_list_skills_reads_frontmatter_only():
    metas = packaged_skills().list_skills()

    dupont = next(m for m in metas if m.name == "dupont-analysis")
    assert dupont.description
    assert "get_indicators" in dupont.allowed_tools
    assert dupont.content_estimate > 0
    # Listing must not require loading the body.
    assert dupont.path.endswith("SKILL.md")


def test_load_returns_body_markdown():
    meta, body = packaged_skills().load("dupont-analysis")

    assert meta.name == "dupont-analysis"
    assert "杜邦" in body
    assert body.startswith("#")


def test_unknown_skill_raises():
    with pytest.raises(SkillError):
        packaged_skills().load("does-not-exist")


def test_missing_directory_yields_an_empty_catalogue(tmp_path):
    library = SkillLibrary(tmp_path / "nope")

    assert library.list_skills() == []
    assert "为空" in library.describe()


def test_list_skills_tool_renders_the_catalogue():
    async def run():
        data = DataAccess([])
        data.settings = Settings()
        return await ListSkillsTool(data).run()

    result = asyncio.run(run())

    assert result.ok is True
    assert "dupont-analysis" in result.content


def test_load_skill_tool_marks_the_context_and_reports_reuse():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    data = DataAccess([])
    data.settings = Settings()

    async def run():
        tool = LoadSkillTool(data, ctx=ctx)
        first = await tool.run(name="dupont-analysis")
        second = await tool.run(name="dupont-analysis")
        return first, second

    first, second = asyncio.run(run())

    assert first.ok is True
    assert "技能：" in first.content
    assert second.ok is True
    assert "复用" in second.content
    assert ctx.loaded_skills == ["dupont-analysis"]


# --- interactive bus ---------------------------------------------------------

def test_confirm_bus_resolves_a_responded_request():
    async def run():
        bus = ConfirmBus(ttl_s=1.0)

        async def answer_soon():
            await asyncio.sleep(0.01)
            pending = bus.pending_ids()
            assert pending, "request should be registered"
            bus.respond(request_id=pending[0], value="y")

        task = asyncio.create_task(answer_soon())
        payload, answer = await bus.request(
            session_id="s", kind="confirm", prompt="run?", options=["y", "n"]
        )
        await task
        return payload, answer

    payload, answer = asyncio.run(run())

    assert payload["kind"] == "confirm"
    assert answer == "y"
    assert payload["request_id"]


def test_confirm_bus_times_out_without_an_answer():
    async def run():
        bus = ConfirmBus(ttl_s=0.05)
        return await bus.request(session_id="s", kind="question", prompt="?") 

    payload, answer = asyncio.run(run())

    assert answer is None
    assert payload["kind"] == "question"


def test_responding_to_an_unknown_request_is_refused():
    bus = ConfirmBus()

    assert bus.respond(request_id="req_missing", value="y") is False


def test_cancel_session_fails_its_pending_requests():
    async def run():
        bus = ConfirmBus(ttl_s=5.0)

        async def cancel_soon():
            await asyncio.sleep(0.01)
            return bus.cancel_session("s")

        task = asyncio.create_task(cancel_soon())
        try:
            await bus.request(session_id="s", kind="confirm", prompt="?")
        except asyncio.CancelledError:
            cancelled = True
        else:  # pragma: no cover
            cancelled = False
        await task
        return cancelled

    assert asyncio.run(run()) is True
