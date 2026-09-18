"""ResearchContext：plan 生命周期、结论与系统注入（文档 03.6.2）。"""

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
    assert second.plan_id == first.plan_id  # 同一个 plan，新的 revision


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


def test_plan_progress_counts_done_steps():
    ctx = make_ctx()
    plan = ctx.set_plan(
        "g", [PlanStep(seq=1, action="a"), PlanStep(seq=2, action="b")]
    )

    assert plan.progress() == (0, 2)
    ctx.mark_plan_step(1, "done")
    assert ctx.plan.progress() == (1, 2)


def test_plan_digest_renders_progress_and_dependency_state():
    ctx = make_ctx()
    ctx.set_plan(
        "分析茅台",
        [
            PlanStep(seq=1, action="取行情", tool_hint=["get_quote"]),
            PlanStep(seq=2, action="算指标", dep=[1]),
        ],
    )
    ctx.mark_plan_step(1, "done")

    digest = ctx.plan_digest()

    # 头部携带 已完成/总数 的 ledger，step 2 显示其依赖项的状态标记。
    assert "进度 1/2" in digest
    assert "依赖：1✓" in digest


def test_add_skill_is_idempotent():
    ctx = make_ctx()

    assert ctx.add_skill("equity-research") is True
    assert ctx.add_skill("equity-research") is False
    assert ctx.loaded_skills == ["equity-research"]


def test_activate_tool_is_idempotent():
    ctx = make_ctx()

    assert ctx.activate_tool("get_announcements") is True
    assert ctx.activate_tool("get_announcements") is False


def test_conclusions_keep_their_cids():
    ctx = make_ctx()

    conclusion = ctx.add_conclusion("ROE 约 30%", ["cit_000001"])

    assert conclusion.cids == ["cit_000001"]
    assert ctx.conclusions[0].text == "ROE 约 30%"
    assert conclusion.ts  # 带时间戳


def test_state_block_carries_only_the_current_date_when_there_is_nothing_else():
    """状态块不再可能为空：当前日期是无条件注入的锚点。

    没有它，模型无法判断某个数据期是否即最新已发布期，也就无法把「月度指标尚未发布」
    与「系统给了旧数据」分开。除此之外，无可报告内容的会话不得凭空多出小节。
    """
    block = make_ctx().state_block()

    assert "当前日期：" in block
    assert "研究计划" not in block


def test_state_block_date_is_the_real_local_date():
    """日期锚点必须是真实日期，否则模型据此判断时效只会得出错误结论。

    这是回答「当前最新的 PMI 是多少」时唯一能说明「9 月数据尚未发布」的依据。
    """
    from datetime import datetime

    today = datetime.now().astimezone().date().isoformat()

    assert f"当前日期：{today}" in make_ctx().state_block()


def test_state_block_includes_plan_skills_and_conclusions():
    ctx = make_ctx()
    ctx.set_plan("分析茅台", [PlanStep(seq=1, action="取行情")])
    ctx.add_skill("equity-research")
    ctx.add_conclusion("ROE 高", ["cit_000001"])

    block = ctx.state_block()

    assert "研究计划" in block
    assert "equity-research" in block
    assert "cit_000001" in block


def test_an_unresolved_review_finding_is_carried_across_turns():
    """终审的未结事项必须持续可见，否则模型可以在后续任一轮把它忘掉（docs 03.10.7）。"""
    ctx = make_ctx()

    ctx.note_review_finding("测试报告", "测试报告：[高] 营收数字缺引用")

    block = ctx.state_block()
    assert "未消解的风险终审问题" in block
    assert "不得声称已复核" in block
    assert "[高] 营收数字缺引用" in block


def test_a_resolved_review_finding_is_dropped_by_topic():
    """修订后重审通过必须能清账，否则一份改好的报告会永远背着旧账。"""
    ctx = make_ctx()
    ctx.note_review_finding("测试报告", "测试报告：[高] 营收数字缺引用")

    ctx.note_review_finding("测试报告", None)

    assert "未消解的风险终审问题" not in ctx.state_block()


def test_multiple_reports_are_tracked_independently():
    ctx = make_ctx()
    ctx.note_review_finding("报告甲", "报告甲：[高] 甲的问题")
    ctx.note_review_finding("报告乙", "报告乙：[高] 乙的问题")

    block = ctx.state_block()

    assert "甲的问题" in block
    assert "乙的问题" in block

    ctx.note_review_finding("报告甲", None)

    block = ctx.state_block()
    assert "甲的问题" not in block
    assert "乙的问题" in block  # 同会话的另一份报告不受影响


def test_note_review_finding_ignores_an_empty_topic():
    ctx = make_ctx()

    ctx.note_review_finding("", "无主题的问题")

    assert "无主题的问题" not in ctx.state_block()


def test_recalled_events_are_capped_by_the_configured_token_budget():
    """recall_max_tokens 过去只存在于配置与文档中，没有任何消费者。

    这里钉住它真正生效：召回注入不得超过该预算，否则一段历史事件可以挤占本应
    留给当前问题的窗口。
    """
    from finharness.context.memory.short_term import Episode, ShortTermMemory

    ctx = ResearchContext(
        cite=CitationRegistry(),
        settings=Settings(context={"recall_max_tokens": 50}),
    )
    memory = ShortTermMemory(cap=50)
    for index in range(20):
        memory.add(
            Episode(
                kind="data",
                subject="600519",
                summary=f"第{index}条召回事件的摘要内容" * 3,
            )
        )
    ctx.short_term = memory
    ctx.recalled = memory.recall(["600519"])

    lines = ctx._recalled_lines()
    text = "\n".join(lines)
    from finharness.context.tokens import default_counter

    assert text
    assert default_counter().count(text).tokens <= 50


def test_recall_injection_has_no_section_header_of_its_own():
    """状态块自己提供小节标题，注入文本不应重复一个。"""
    from finharness.context.memory.short_term import Episode, ShortTermMemory

    ctx = ResearchContext(cite=CitationRegistry(), settings=Settings())
    memory = ShortTermMemory(cap=10)
    memory.add(Episode(kind="data", subject="600519", summary="摘要"))
    ctx.short_term = memory
    ctx.recalled = memory.recall(["600519"])

    block = ctx.state_block()

    assert block.count("相关历史事件") == 1
