"""通用 sub-agent fan-out 的测试：spawn_agent（文档 03.10）。

risk reviewer 有自己的测试套件（test_reviewer.py）；这些测试覆盖通用机制——
N 个隔离任务输入，N 个结论输出——以及只有 sub-agent 并发运行时才重要的特性。
"""

from __future__ import annotations

import asyncio

import pandas as pd

from finharness.config.settings import ContextSettings, Settings
from finharness.coordinator import GENERAL_FOCUS, MAX_SPAWN_TASKS, READER_FOCUS, Coordinator
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.cost import SessionStats
from finharness.provider.base import Provider
from finharness.tools.registry import reader_tool_names
from finharness.types import ModelUsage, StreamChunk, StreamEvent, ToolUse
from tests.conftest import settings_with_cache


class Adapter(DataAdapter):
    """worker 绝不能访问的数据源。"""

    name = "fake"

    def fetch_quote(self, symbol):
        return FetchResult(
            df=pd.DataFrame([{"symbol": symbol, "close": 100.0}]),
            interface="fake_quote",
        )


def message_end(*tool_uses: ToolUse) -> StreamChunk:
    return StreamChunk(
        StreamEvent.MESSAGE_END,
        ModelUsage(input_tokens=5, output_tokens=2, tool_uses=list(tool_uses)),
    )


def text_round(*parts: str) -> list[StreamChunk]:
    return [StreamChunk(StreamEvent.TEXT_DELTA, part) for part in parts] + [message_end()]


class EchoTaskProvider(Provider):
    """用各自的任务文本答复每个 sub-agent，使结果可追溯。

    并发安全：在列表中记录请求，且不假设顺序。
    """

    def __init__(self, *, fail_on: str | None = None):
        self.fail_on = fail_on
        self.requests: list[dict] = []

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
        names = [entry["function"]["name"] for entry in tools]
        self.requests.append({"system": system, "messages": list(messages), "tools": names})
        # 任务就是唯一的 user 消息；将其回显作为结论。
        text = ""
        for msg in messages:
            if getattr(msg, "role", None) == "user" and msg.content:
                text = msg.content
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("provider boom")
        yield StreamChunk(StreamEvent.TEXT_DELTA, f"结论：{text.strip()[:40]}")
        yield message_end()


class FetchingProvider(Provider):
    """在给出结论前发出一次 tool call，以检验 tool loop。"""

    def __init__(self, tool_name: str, args: dict):
        self.tool_name = tool_name
        self.args = args
        self.rounds = 0

    async def stream(self, *, system: str, messages: list, tools: list[dict], usage: ModelUsage):
        self.rounds += 1
        if self.rounds == 1:
            yield message_end(ToolUse(call_id="c1", name=self.tool_name, args=self.args))
            return
        yield StreamChunk(StreamEvent.TEXT_DELTA, "读完了")
        yield message_end()


def make_settings(tmp_path, **context) -> Settings:
    return settings_with_cache(
        tmp_path,
        context=ContextSettings(**context),
        paths={"output_dir": tmp_path / "output"},
    )


def make_coordinator(tmp_path, provider, *, cite=None, stats=None, usage=None, parent_gate=None):
    settings = make_settings(tmp_path)
    data = DataAccess([Adapter()], cache=LocalCache(tmp_path / "cache"), settings=settings)
    coordinator = Coordinator(
        provider=provider,
        data=data,
        settings=settings,
        cite=cite if cite is not None else CitationRegistry(),
        parent_gate=parent_gate,
    )
    if stats is not None or usage is not None:
        coordinator.bind_accounting(
            usage=usage if usage is not None else ModelUsage(),
            stats=stats if stats is not None else SessionStats(),
        )
    return coordinator


def run(coro):
    return asyncio.run(coro)


# -- fan-out 形状 ------------------------------------------------------------


def test_spawn_reports_started_and_completed_for_each_task(tmp_path):
    """主界面要用现有 tool_progress 通道展示子任务起止，不能等整次 spawn 结束。"""
    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider)
    seen: list[tuple[int, int, str, str]] = []

    async def on_task(index: int, total: int, task: str, status: str) -> None:
        seen.append((index, total, task, status))

    results = run(
        coordinator.spawn(tasks=["任务甲", "任务乙"], on_task=on_task)
    )

    assert [item.ok for item in results] == [True, True]
    starts = [item for item in seen if item[3] == "started"]
    ends = [item for item in seen if item[3] in {"completed", "failed"}]
    assert {(i, t, task) for i, t, task, _ in starts} == {
        (0, 2, "任务甲"),
        (1, 2, "任务乙"),
    }
    assert {(i, t, task) for i, t, task, _ in ends} == {
        (0, 2, "任务甲"),
        (1, 2, "任务乙"),
    }
    assert all(status == "completed" for *_, status in ends)


def test_each_task_yields_its_own_result_in_order(tmp_path):
    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider)

    results = run(coordinator.spawn(tasks=["任务甲", "任务乙", "任务丙"]))

    assert [item.focus for item in results] == [GENERAL_FOCUS] * 3
    assert len(results) == 3
    # 顺序与输入一致，调用方可将结果 zip 回任务。
    assert [item.task for item in results] == ["任务甲", "任务乙", "任务丙"]
    assert all(item.ok for item in results)
    assert all("结论" in item.summary for item in results)


def test_one_call_runs_every_task(tmp_path):
    """Fan-out 是内部的：N 个任务不应依赖 N 次模型调用。"""
    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider)

    results = run(coordinator.spawn(tasks=["甲", "乙", "丙", "丁"]))

    assert len(results) == 4
    # 每个任务一个 sub-agent，因此恰好四次 provider 请求。
    assert len(provider.requests) == 4


# -- 隔离 ---------------------------------------------------------------------


def test_workers_see_only_their_own_task_and_no_main_transcript(tmp_path):
    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider)

    run(coordinator.spawn(tasks=["任务甲", "任务乙"], context="共享背景"))

    assert len(provider.requests) == 2
    for request in provider.requests:
        user_messages = [m for m in request["messages"] if getattr(m, "role", None) == "user"]
        # 恰好一条 user 消息：任务加上共享背景。不会转发主 transcript。
        assert len(user_messages) == 1
        body = user_messages[0].content
        assert "共享背景" in body
        # 该 agent 的上下文只提及两个任务中的一个，绝不同时提及两者——
        # 兄弟任务不得泄漏进来。
        assert sum(task in body for task in ("任务甲", "任务乙")) == 1


def test_the_general_sub_agent_can_reach_read_only_data_tools(tmp_path):
    """通用子代理被派去"分析某实体"时要能自己取数：其工具集即只读取数子集。

    仍然只读——不含写工具、不含 META（因此结构上不能再派生、不能改任何东西）。
    任务要求检索公开网页时可以 web_search；risk 复核者仍然不能联网。
    """
    from finharness.tools.registry import general_tool_names

    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider)

    run(coordinator.spawn(tasks=["分析茅台(600519)的盈利能力"]))

    tools = set(provider.requests[0]["tools"])
    assert tools == set(general_tool_names())
    # 可取数，任务要求时也可联网。
    assert "get_quote" in tools and "get_indicators" in tools
    assert "web_search" in tools
    # 但不能写、不能派生、不能影响会话状态。
    for name in ("write_file", "spawn_agent", "search_tools", "ask_user", "remember_preference"):
        assert name not in tools


def test_the_reader_sub_agent_gets_local_material_tools_only(tmp_path):
    """内部 reader 只消化材料、不取数——这是结构性边界。"""
    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider)

    run(coordinator.spawn(tasks=["摘要这段材料"], focus=READER_FOCUS))

    tools = set(provider.requests[0]["tools"])
    assert tools == set(reader_tool_names())
    assert "get_quote" not in tools and "get_research_reports" not in tools


def test_a_reader_cannot_reach_the_data_layer_even_if_asked(tmp_path):
    """reader 不取数是结构性的而非指令性的：该名称根本无法解析。"""
    provider = FetchingProvider("get_quote", {"symbol": "600519"})
    coordinator = make_coordinator(tmp_path, provider)

    results = run(coordinator.spawn(tasks=["查一下茅台股价"], focus=READER_FOCUS))

    # tool call 因未知而被拒绝；sub-agent 仍会给出结论。
    assert results[0].ok is True
    assert results[0].citations == []


# -- 失败隔离 -----------------------------------------------------------------


def test_one_failing_task_does_not_sink_its_siblings(tmp_path):
    """崩溃的 sub-agent 只是一行坏结果，而非整批失败。"""
    provider = EchoTaskProvider(fail_on="坏任务")
    coordinator = make_coordinator(tmp_path, provider)

    results = run(coordinator.spawn(tasks=["好任务一", "坏任务", "好任务二"]))

    assert len(results) == 3
    assert [item.ok for item in results] == [True, False, True]
    assert results[1].error is not None
    assert "好任务" in results[0].summary


def test_spawn_never_raises_on_a_provider_failure(tmp_path):
    provider = EchoTaskProvider(fail_on="都失败")
    coordinator = make_coordinator(tmp_path, provider)

    results = run(coordinator.spawn(tasks=["都失败1", "都失败2"]))

    assert all(item.ok is False for item in results)
    assert all(item.error for item in results)


# -- 并发下的 citation 归因 ---------------------------------------------------


def test_concurrent_workers_attribute_citations_exactly(tmp_path):
    """并发陷阱：N 个 sub-agent 不得相互认领对方的 cid。

    对共享 registry 做前后 diff 会让每个 worker 得到相同的集合；
    而 scoped registry 只记录各自铸造的内容。
    """
    shared = CitationRegistry()
    provider = FetchingProvider("read_file", {"path": "output/whatever.md"})
    coordinator = make_coordinator(tmp_path, provider, cite=shared)

    results = run(coordinator.spawn(tasks=["读甲", "读乙", "读丙"]))

    claimed: list[str] = []
    for item in results:
        claimed.extend(item.citations)
    # 每个被认领的 cid 都是真实的，且没有重复认领——归因精确。
    assert len(claimed) == len(set(claimed))
    assert all(shared.get(cid) is not None for cid in claimed)
    assert len(shared.all()) == len(claimed)


# -- 账目统计 -----------------------------------------------------------------


def test_spawn_folds_every_sub_agent_into_the_focus_breakdown(tmp_path):
    usage = ModelUsage()
    stats = SessionStats()
    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider, stats=stats, usage=usage)

    run(coordinator.spawn(tasks=["甲", "乙", "丙"]))

    assert stats.snapshot().per_agent[GENERAL_FOCUS]["runs"] == 3
    assert usage.input_tokens == 15  # 3 个 agent x 5 个 input tokens
    # worker 的读取不得计入主 agent 的 tool 使用。
    assert "read_file" not in stats.snapshot().per_tool


# -- 输入校验 -----------------------------------------------------------------


def test_an_empty_task_list_is_refused(tmp_path):
    coordinator = make_coordinator(tmp_path, EchoTaskProvider())

    for tasks in ([], ["   "], ["", "  "]):
        try:
            run(coordinator.spawn(tasks=tasks))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {tasks!r}")


def test_too_many_tasks_are_refused(tmp_path):
    coordinator = make_coordinator(tmp_path, EchoTaskProvider())

    try:
        run(coordinator.spawn(tasks=[f"任务{i}" for i in range(MAX_SPAWN_TASKS + 1)]))
    except ValueError as exc:
        assert "最多" in str(exc)
        return
    raise AssertionError("expected ValueError for too many tasks")


def test_an_unknown_focus_is_refused(tmp_path):
    coordinator = make_coordinator(tmp_path, EchoTaskProvider())

    try:
        run(coordinator.spawn(tasks=["甲"], focus="no_such_focus"))
    except ValueError as exc:
        assert "未知" in str(exc)
        return
    raise AssertionError("expected ValueError for an unknown focus")


def test_spawn_confirms_egress_once_before_dispatching_general_workers(tmp_path):
    """general worker 能 web_search，派发前须走一次与主会话同类的 egress 确认。"""
    from finharness.permissions.gate import PermissionGate

    asked: list[str] = []

    async def confirm(name, args):
        asked.append(name)
        return True

    settings = make_settings(tmp_path)
    gate = PermissionGate(settings=settings, confirm_egress=confirm)
    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider, parent_gate=gate)

    run(coordinator.spawn(tasks=["搜索锂电池产能过剩的公开讨论"]))
    assert asked == ["web_search"]
    asked.clear()
    run(coordinator.spawn(tasks=["再搜一则交叉印证"]))
    assert asked == []


def test_denied_egress_does_not_dispatch_general_workers(tmp_path):
    from finharness.permissions.gate import PermissionGate

    async def confirm(name, args):
        return False

    settings = make_settings(tmp_path)
    gate = PermissionGate(settings=settings, confirm_egress=confirm)
    provider = EchoTaskProvider()
    coordinator = make_coordinator(tmp_path, provider, parent_gate=gate)

    results = run(coordinator.spawn(tasks=["搜索公开讨论"]))

    assert results[0].ok is False
    assert results[0].error
    assert provider.requests == []
