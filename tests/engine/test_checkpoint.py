"""断点与恢复（docs 03.3）。

覆盖三件互相独立的事：

* 硬取消（连接被掐断、任务被取消）也要落库——`CancelledError` 不是 `Exception`，
  若不在 `_run_once` 兜住，本轮成果会静默消失；
* `Plan` 的序列化往返保真（plan_id/revision 原样），因为恢复的语义是
  "在同一份计划上继续"；
* 恢复时计划被装回 `ctx`，且旧断点被消费（不会在运行途中仍显示"可继续"）。
"""

import asyncio
import sys
from pathlib import Path

import pytest

from finharness.config.settings import ContextSettings, Settings
from finharness.context.memory.store import MemoryStore
from finharness.context.session import Plan, PlanStep, ResearchContext
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.provider.base import Provider
from finharness.types import ModelUsage, StreamChunk, StreamEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from test_loop import (  # noqa: E402
    ScriptedProvider,
    StubRegistry,
    message_end,
    text_round,
)


def make_settings(tmp_path) -> Settings:
    return Settings(
        context=ContextSettings(context_window_tokens=100000),
        data={"cache_dir": tmp_path / "cache"},
    )


def build_loop(tmp_path, provider, *, store, conversation_id, ctx=None):
    settings = make_settings(tmp_path)
    cite = CitationRegistry()
    ctx = ctx or ResearchContext(cite=cite, settings=settings)
    return AgentLoop(
        provider=provider,
        registry=StubRegistry(),
        settings=settings,
        system="系统提示",
        cite=cite,
        ctx=ctx,
        conversation_id=conversation_id,
        store=store,
    )


# --- Plan 往返 -----------------------------------------------------------------

def test_plan_roundtrip_preserves_identity_and_revision():
    plan = Plan(
        plan_id="plan_004",
        goal="研究贵州茅台",
        steps=[
            PlanStep(seq=1, action="取财务数据", tool_hint=["get_financials"], status="done"),
            PlanStep(seq=2, action="估值", dep=[1]),
        ],
        revision=3,
    )

    restored = Plan.from_dict(plan.to_dict())

    assert restored is not None
    assert restored.plan_id == "plan_004"
    assert restored.revision == 3
    assert restored.goal == "研究贵州茅台"
    assert [step.status for step in restored.steps] == ["done", "pending"]
    assert restored.steps[1].dep == [1]
    assert restored.steps[0].tool_hint == ["get_financials"]


def test_plan_from_dict_rejects_empty_and_sanitizes_damage():
    # 空计划没有恢复价值：宁可不恢复，也不要装上一份空壳。
    assert Plan.from_dict({"steps": []}) is None
    assert Plan.from_dict({}) is None

    # 损坏的字段降级为默认值，而不是让恢复本身抛错。
    damaged = Plan.from_dict(
        {"steps": [{"seq": "nonsense", "status": "made-up", "dep": ["x", 2]}]}
    )
    assert damaged is not None
    assert damaged.steps[0].seq == 0
    assert damaged.steps[0].status == "pending"
    assert damaged.steps[0].dep == [2]


# --- 硬取消也要落库 ------------------------------------------------------------

def test_hard_cancel_still_persists_the_turn(tmp_path):
    """任务被取消（而非优雅停止）时，本轮提问与已产出的消息仍必须落库。"""
    store = MemoryStore(tmp_path / "memory.db")
    started = asyncio.Event()

    class StallingProvider(Provider):
        async def stream(self, *, system, messages, tools, usage):
            # 产出一段 delta 后永久挂起，模拟"用户停止后连接被硬掐断"。
            yield StreamChunk(StreamEvent.TEXT_DELTA, "半截回答")
            started.set()
            await asyncio.sleep(30)
            yield message_end()

    loop = build_loop(
        tmp_path, StallingProvider(), store=store, conversation_id="c_cancel"
    )

    async def run():
        task = asyncio.create_task(loop.run("被中断的问题"))
        await started.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    # 提问落库了：否则这条对话在存储里会像从未发生过一样。
    stored = store.load_messages("c_cancel")
    assert any(message.content == "被中断的问题" for message in stored)
    # 硬取消被记为"被中断"，同样留下断点而非无痕。
    checkpoint = store.load_latest_checkpoint("c_cancel")
    assert checkpoint is not None
    assert checkpoint.status == "stopped"
    assert checkpoint.reason == "interrupted"


# --- 恢复 ----------------------------------------------------------------------

def test_restore_plan_from_stopped_checkpoint(tmp_path):
    """上一轮被停止后，新一轮应把原计划装回 ctx（并消费掉该断点）。"""
    store = MemoryStore(tmp_path / "memory.db")
    conversation_id = "c_resume"
    plan = Plan(
        plan_id="plan_007",
        goal="完成深度研究",
        steps=[PlanStep(seq=1, action="取数", status="done")],
        revision=2,
    )
    store.ensure_conversation(conversation_id, user_id="", title="t")
    store.save_checkpoint(
        conversation_id,
        status="stopped",
        reason="user_stopped",
        rounds=4,
        plan=plan.to_dict(),
        partial_answer="部分结论",
    )

    loop = build_loop(
        tmp_path, ScriptedProvider([text_round("继续后的答案")]), store=store,
        conversation_id=conversation_id,
    )

    async def run():
        return await loop.run("继续")

    asyncio.run(run())

    # 计划在载入内存时被恢复——这是"不重跑已完成步骤"的依据。
    # （运行结束后 ctx 里是我们恢复的那份，修订号保持。）
    assert loop.ctx.plan is not None
    assert loop.ctx.plan.plan_id == "plan_007"
    assert loop.ctx.plan.revision >= 2


def test_completed_checkpoint_does_not_leak_plan_into_new_turn(tmp_path):
    """已交付完毕的轮次不提供"继续"：它的计划不该被下一轮继承。"""
    store = MemoryStore(tmp_path / "memory.db")
    conversation_id = "c_leak"
    plan = Plan(plan_id="plan_009", goal="旧目标", steps=[PlanStep(seq=1, action="x")])
    store.ensure_conversation(conversation_id, user_id="", title="t")
    store.save_checkpoint(conversation_id, status="completed", plan=plan.to_dict())

    loop = build_loop(
        tmp_path, ScriptedProvider([text_round("新答案")]), store=store,
        conversation_id=conversation_id,
    )

    async def run():
        return await loop.run("一个全新的问题")

    asyncio.run(run())

    # 新问题从零开始，没有计划。
    assert loop.ctx.plan is None


def test_restore_consumes_checkpoint_before_run(tmp_path):
    """新一轮一旦开始，旧的"可继续"就该失效，界面不该在运行途中仍显示续做入口。"""
    store = MemoryStore(tmp_path / "memory.db")
    conversation_id = "c_consume"
    plan = Plan(plan_id="plan_011", goal="g", steps=[PlanStep(seq=1, action="a")])
    store.ensure_conversation(conversation_id, user_id="", title="t")
    store.save_checkpoint(conversation_id, status="stopped", plan=plan.to_dict())

    loop = build_loop(
        tmp_path, ScriptedProvider([text_round("答案")]), store=store,
        conversation_id=conversation_id,
    )

    async def run():
        return await loop.run("继续")

    asyncio.run(run())

    # 运行收尾会写入属于它自己的新断点（completed），旧的 stopped 已被消费。
    checkpoint = store.load_latest_checkpoint(conversation_id)
    assert checkpoint is not None
    assert checkpoint.status == "completed"


def test_every_run_consumes_the_previous_checkpoint_even_when_memory_is_loaded(tmp_path):
    """会话复用时，新一轮也必须**消费**掉旧断点（回归）。

    断点属于"一轮"，而一个会话可承载多轮。恢复/消费被挂在"会话首轮加载记忆"
    里时，会话被复用时那段逻辑会提前返回，于是旧断点在整个新一轮运行期间都
    仍然可见——对外表现为：模型正在为新一轮工作时，服务端仍在报告"上一轮
    可继续"。把处理移进每次 `run()` 的入口即可保证"本轮一开始就作废旧断点"。

    断言取运行**中途**的存储状态，因为收尾写入会覆盖断点，掩盖这个差异。
    """
    store = MemoryStore(tmp_path / "memory.db")
    conversation_id = "c_consume_reuse"
    loop = build_loop(
        tmp_path, ScriptedProvider([text_round("第一轮答案")]), store=store,
        conversation_id=conversation_id,
    )

    async def first():
        return await loop.run("第一轮问题")

    asyncio.run(first())
    # 会话被复用：记忆已加载过，因此首轮加载路径不会再跑。
    assert loop._memory_loaded is True
    store.save_checkpoint(
        conversation_id, status="stopped", reason="user_stopped",
        plan=Plan(plan_id="plan_031", goal="旧目标", steps=[PlanStep(seq=1, action="a")]).to_dict(),
    )

    seen_mid_run: dict = {}

    class InspectingProvider(ScriptedProvider):
        async def stream(self, *, system, messages, tools, usage):
            # 运行途中读一次：此刻旧断点应当已被消费。
            seen_mid_run["checkpoint"] = store.load_latest_checkpoint(conversation_id)
            async for chunk in super().stream(
                system=system, messages=messages, tools=tools, usage=usage
            ):
                yield chunk

    loop.provider = InspectingProvider([text_round("第二轮答案")])

    async def second():
        return await loop.run("继续")

    asyncio.run(second())

    assert seen_mid_run["checkpoint"] is None, (
        "新一轮运行途中旧断点仍可见——模型在为新请求工作时，界面可能仍显示"
        "上一轮的'继续研究'入口"
    )
