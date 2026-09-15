"""Skill 目录、meta 工具与交互总线。"""

import asyncio

import pytest

from finharness.config.settings import Settings
from finharness.context.session import PlanStep, ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.server.confirm import ConfirmBus
from finharness.tools.meta.plan import RecordConclusionTool, UpdatePlanStepTool
from finharness.tools.meta.skills import (
    ListSkillsTool,
    LoadSkillTool,
    SkillError,
    SkillRegistry,
)


def packaged_skills() -> SkillRegistry:
    return SkillRegistry(Settings().paths.skills_dir)


def test_packaged_catalogue_lists_the_scenarios():
    names = {meta.name for meta in packaged_skills().list_skills()}
    assert {"equity-research", "industry-research", "macro-research", "quant-factor"} <= names


def test_list_skills_reads_frontmatter_only():
    metas = packaged_skills().list_skills()

    equity = next(m for m in metas if m.name == "equity-research")
    assert equity.description
    assert "get_financials" in equity.allowed_tools
    assert equity.content_estimate > 0
    # 列出目录不得要求加载正文。
    assert equity.path.endswith("SKILL.md")


def test_load_returns_body_markdown():
    meta, body, record = packaged_skills().load("equity-research")

    assert meta.name == "equity-research"
    assert "个股" in body
    assert body.startswith("#")
    assert record.reused is False


def test_unknown_skill_raises():
    with pytest.raises(SkillError):
        packaged_skills().load("does-not-exist")


def test_missing_directory_yields_an_empty_catalogue(tmp_path):
    library = SkillRegistry(tmp_path / "nope")

    assert library.list_skills() == []
    assert "为空" in library.describe()


def test_list_skills_tool_renders_the_catalogue():
    async def run():
        data = DataAccess([])
        data.settings = Settings()
        return await ListSkillsTool(data).run()

    result = asyncio.run(run())

    assert result.ok is True
    assert "equity-research" in result.content


def test_load_skill_tool_marks_the_context_and_reports_reuse():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    data = DataAccess([])
    data.settings = Settings()

    async def run():
        tool = LoadSkillTool(data, ctx=ctx)
        first = await tool.run(name="equity-research")
        second = await tool.run(name="equity-research")
        return first, second

    first, second = asyncio.run(run())

    assert first.ok is True
    assert "技能：" in first.content
    assert second.ok is True
    assert "复用" in second.content
    assert ctx.loaded_skills == ["equity-research"]


# --- 计划回写工具（文档 03.3.9）----------------------------------------------

def test_update_plan_step_writes_status_and_reports_digest():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    ctx.set_plan("g", [PlanStep(seq=1, action="取行情"), PlanStep(seq=2, action="算指标")])
    data = DataAccess([])
    data.settings = Settings()

    async def run():
        tool = UpdatePlanStepTool(data, ctx=ctx)
        return await tool.run(seq=1, status="done")

    result = asyncio.run(run())

    assert result.ok is True
    assert ctx.plan.steps[0].status == "done"
    assert "进度 1/2" in result.content


def test_update_plan_step_rejects_an_unknown_seq():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    ctx.set_plan("g", [PlanStep(seq=1, action="a")])
    data = DataAccess([])
    data.settings = Settings()

    async def run():
        return await UpdatePlanStepTool(data, ctx=ctx).run(seq=99, status="done")

    result = asyncio.run(run())

    # ValueError 会被转换为结构化失败，而非抛出异常。
    assert result.ok is False
    assert "99" in (result.error or "")


def test_record_conclusion_persists_text_and_cids():
    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    data = DataAccess([])
    data.settings = Settings()

    async def run():
        return await RecordConclusionTool(data, ctx=ctx).run(
            text="ROE 约 30%", cids=["cit_000001"]
        )

    result = asyncio.run(run())

    assert result.ok is True
    assert ctx.conclusions[0].text == "ROE 约 30%"
    assert ctx.conclusions[0].cids == ["cit_000001"]
    assert "ROE 约 30%" in result.content


# --- 交互总线 ----------------------------------------------------------------

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
