"""通过 HTTP 恢复对话：列出、回放并继续。

session 是执行窗口，会过期；conversation 是记忆作用域，其生命周期更长。
这些测试覆盖客户端可见的结果：列出 endpoint、对话记录回放，以及按
conversation id 继续时会复用同一份记忆，而不是从头开始。
"""

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app
from finharness.types import (
    STATE_VIEW_META,
    ModelUsage,
    Msg,
    StreamChunk,
    StreamEvent,
    ToolUse,
)


def _is_user_turn(message: Msg) -> bool:
    """一条真实的用户提问；排除随请求追加的会话研究状态视图。

    状态视图以 user 角色发送（放在历史之后以利缓存与注意力），但它不是用户轮次——
    若计入，「恢复历史」类断言会把状态块误当成一问。
    """
    return message.role == "user" and not message.metadata.get(STATE_VIEW_META)


class EchoProvider(FakeProvider):
    """回显此前的用户轮次数量，以暴露被恢复的历史。"""

    def __init__(self):
        super().__init__([])
        self.seen_user_turns: list[int] = []

    async def stream(self, *, system, messages, tools, usage: ModelUsage):
        self.requests.append(list(messages))
        users = [m for m in messages if _is_user_turn(m)]
        self.seen_user_turns.append(len(users))
        yield StreamChunk(StreamEvent.TEXT_DELTA, f"第{len(users)}问已回答")
        yield StreamChunk(
            StreamEvent.MESSAGE_END, ModelUsage(input_tokens=1, output_tokens=1)
        )


class PlannedProvider(EchoProvider):
    """在返回最终答案之前运行一个规划工具。"""

    async def stream(self, *, system, messages, tools, usage: ModelUsage):
        if not any(message.role == "tool_result" for message in messages):
            yield StreamChunk(
                StreamEvent.MESSAGE_END,
                ModelUsage(
                    input_tokens=2,
                    output_tokens=1,
                    tool_uses=[
                        ToolUse(
                            "plan-1",
                            "research_plan",
                            {
                                "goal": "完成公司研究",
                                "steps": [
                                    {
                                        "seq": 1,
                                        "action": "分析财务数据",
                                        "tool_hint": [],
                                        "skill_hint": [],
                                        "dep": [],
                                    }
                                ],
                            },
                        )
                    ],
                ),
            )
            return
        yield StreamChunk(StreamEvent.TEXT_DELTA, "规划任务已完成")
        yield StreamChunk(
            StreamEvent.MESSAGE_END, ModelUsage(input_tokens=3, output_tokens=2)
        )


class ChartProvider(EchoProvider):
    """先画一张图，再给出最终答案，用于验证产出文件能否被回放。"""

    async def stream(self, *, system, messages, tools, usage: ModelUsage):
        if not any(message.role == "tool_result" for message in messages):
            yield StreamChunk(
                StreamEvent.MESSAGE_END,
                ModelUsage(
                    input_tokens=1,
                    output_tokens=1,
                    tool_uses=[
                        ToolUse(
                            "chart-1",
                            "make_chart",
                            {"symbol": "600519", "title": "回放测试走势", "type": "line"},
                        )
                    ],
                ),
            )
            return
        yield StreamChunk(StreamEvent.TEXT_DELTA, "图表已生成")
        yield StreamChunk(
            StreamEvent.MESSAGE_END, ModelUsage(input_tokens=1, output_tokens=1)
        )


class OfflineKlineAdapter(DataAdapter):
    """离线 K 线来源，使 make_chart 无需网络即可产出 PNG。"""

    name = "offline"

    def fetch_kline(self, symbol, period, adjust, years):
        dates = pd.date_range("2025-01-01", periods=40, freq="D")
        return FetchResult(
            df=pd.DataFrame({"date": dates, "close": [100.0 + i for i in range(40)]}),
            interface="offline_kline",
        )


def make_client(
    tmp_path, provider: EchoProvider | None = None, *, adapters: list | None = None
) -> tuple[TestClient, EchoProvider]:
    provider = provider or EchoProvider()
    settings = Settings(
        paths={"memory_db": tmp_path / "memory.db", "output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )
    data_access = DataAccess(adapters or [], settings=settings) if adapters else None
    from tests.server.conftest import authed_client

    client = authed_client(
        TestClient(
            create_app(provider=provider, data_access=data_access, settings=settings, single_tenant=True)
        )
    )
    return (client, provider)


def conversation_id_from(text: str) -> str:
    return text.split('"conversation_id": "', 1)[1].split('"', 1)[0]


def test_first_turn_allocates_a_conversation_id(tmp_path):
    client, _ = make_client(tmp_path)

    response = client.post("/v1/chat/stream", json={"message": "第一问"})

    assert response.status_code == 200
    assert '"conversation_id": "c_' in response.text


def test_conversation_is_listed_after_a_turn(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/v1/chat/stream", json={"message": "茅台分析"})

    body = client.get("/v1/conversations").json()

    assert body["conversations"], "a conversation should be listed"
    assert body["conversations"][0]["title"] == "茅台分析"


def test_transcript_can_be_replayed(tmp_path):
    client, _ = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "第一问"})
    cid = conversation_id_from(first.text)

    body = client.get(f"/v1/conversations/{cid}/messages").json()

    roles = [(m["role"], m["text"]) for m in body["messages"]]
    assert ("user", "第一问") in roles
    assert any(role == "assistant" for role, _ in roles)


def test_replay_keeps_the_turn_trace_and_metrics_after_refresh(tmp_path):
    """重新加载已存储的答案时，不得丢弃其可观察的运行元数据。"""
    client, _ = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "第一问"})
    cid = conversation_id_from(first.text)

    messages = client.get(f"/v1/conversations/{cid}/messages").json()["messages"]
    answer = next(message for message in messages if message["role"] == "assistant")

    turn = answer["turn"]
    assert turn["events"][-1]["event"] == "done"
    usage = turn["events"][-1]["data"]["usage"]
    assert usage["input_tokens"] == 1
    assert usage["output_tokens"] == 1
    assert turn["first_token_ms"] >= 0
    assert turn["total_duration_ms"] >= turn["first_token_ms"]


def test_replay_keeps_planning_tool_events_after_refresh(tmp_path):
    client, _ = make_client(tmp_path, PlannedProvider())
    first = client.post("/v1/chat/stream", json={"message": "执行复杂研究"})
    cid = conversation_id_from(first.text)

    messages = client.get(f"/v1/conversations/{cid}/messages").json()["messages"]
    answer = next(message for message in messages if message["role"] == "assistant")
    tool_events = [
        event
        for event in answer["turn"]["events"]
        if event["event"] == "tool_status"
    ]

    assert [(event["data"]["name"], event["data"]["status"]) for event in tool_events] == [
        ("research_plan", "started"),
        ("research_plan", "completed"),
    ]

    progress = next(
        event["data"]
        for event in answer["turn"]["events"]
        if event["event"] == "plan_progress"
    )
    assert progress["goal"] == "完成公司研究"
    assert progress["steps"] == [
        {"seq": 1, "action": "分析财务数据", "status": "pending", "dep": []}
    ]
    assert "tool_hint" not in progress["steps"][0]


def test_replay_keeps_produced_files_after_refresh(tmp_path):
    """刷新后“产出文件”一栏不得消失。

    客户端只在内存里保存本轮的产出文件，页面刷新会丢掉它们；
    因此这些路径必须随轮次事件持久化，并在回放时原样返回，
    前端才能据此重建下载链接。
    """
    client, _ = make_client(
        tmp_path, ChartProvider(), adapters=[OfflineKlineAdapter()]
    )
    first = client.post("/v1/chat/stream", json={"message": "画一张走势图"})
    cid = conversation_id_from(first.text)

    messages = client.get(f"/v1/conversations/{cid}/messages").json()["messages"]
    answer = next(message for message in messages if message["role"] == "assistant")
    completed = [
        event["data"]
        for event in answer["turn"]["events"]
        if event["event"] == "tool_status" and event["data"]["status"] == "completed"
    ]

    attachments = [path for data in completed for path in data.get("attachments", [])]
    assert attachments, "the produced file must survive into the replayed turn"
    assert any(path.endswith(".png") for path in attachments)
    assert all(
        Path(path).is_file() for path in attachments
    ), "replayed paths must still be downloadable"


def test_replay_omits_tool_frames(tmp_path):
    """只返回可读的回合；工作状态不展示给阅读者。"""
    client, _ = make_client(tmp_path)
    # 直接在 store 中种入一个包含工具轮次的对话（归属当前测试用户）。
    store = client.app.state.memory_store
    store.ensure_conversation("c_tools", user_id=client.finharness_user["id"])
    store.append_messages(
        "c_tools",
        [
            Msg.user("查报价"),
            Msg(
                role="assistant",
                content=None,
                tool_uses=[ToolUse("c1", "get_quote", {})],
            ),
            Msg(role="tool_result", content=None, tool_results=[("c1", '{"ok":true}')]),
            Msg(role="assistant", content="报价是 100"),
        ],
    )

    body = client.get("/v1/conversations/c_tools/messages").json()

    assert body["messages"] == [
        {"role": "user", "text": "查报价"},
        {"role": "assistant", "text": "报价是 100"},
    ]


def test_replay_404s_for_an_unknown_conversation(tmp_path):
    client, _ = make_client(tmp_path)

    assert client.get("/v1/conversations/c_missing/messages").status_code == 404


def test_continuing_by_conversation_id_restores_history(tmp_path):
    """核心验收点：恢复对话必须让模型看到更早的轮次。"""
    client, provider = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "第一问"})
    cid = conversation_id_from(first.text)

    client.post("/v1/chat/stream", json={"conversation_id": cid, "message": "第二问"})

    # 第二个请求应携带两个用户轮次，而不只是新的那一轮。
    assert provider.seen_user_turns[-1] == 2


def test_a_new_conversation_starts_with_no_history(tmp_path):
    client, provider = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "甲对话的问题"})
    assert first.status_code == 200

    # 不带 conversation id 的请求是一个全新对话。
    client.post("/v1/chat/stream", json={"message": "乙对话的问题"})

    assert provider.seen_user_turns[-1] == 1
    conversations = client.get("/v1/conversations").json()["conversations"]
    assert len(conversations) == 2, "each conversation is tracked separately"


def test_resuming_after_the_session_expired_still_restores_history(tmp_path):
    """conversation id 的生命周期长于执行窗口；这正是它们存在的意义。"""
    client, provider = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "第一问"})
    cid = conversation_id_from(first.text)

    # 让活动 session 过期，同时不触碰已存储的 conversation。
    registry = client.app.state.session_registry
    for session in registry.sessions.values():
        session.last_active -= 10_000

    client.post("/v1/chat/stream", json={"conversation_id": cid, "message": "第二问"})

    assert provider.seen_user_turns[-1] == 2, "memory must outlive the session"


def test_citations_can_be_read_by_conversation(tmp_path):
    client, _ = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "第一问"})
    cid = conversation_id_from(first.text)

    response = client.get("/v1/citations", params={"conversation_id": cid})

    assert response.status_code == 200
    assert response.json()["count"] == 0  # 没有工具运行，但作用域可以解析


def test_citations_404_for_an_unknown_conversation(tmp_path):
    client, _ = make_client(tmp_path)

    assert (
        client.get("/v1/citations", params={"conversation_id": "c_none"}).status_code
        == 404
    )


def test_persisted_citations_are_readable_after_the_session_is_gone(tmp_path):
    """conversation 的生命周期长于进程；其来源必须保持可寻址。"""
    from finharness.data.citation import Citation

    client, _ = make_client(tmp_path)
    store = client.app.state.memory_store
    store.ensure_conversation("c_sources", user_id=client.finharness_user["id"])
    store.save_citations(
        "c_sources",
        [
            Citation(
                cid="cit_000001",
                tool="get_quote",
                endpoint="akshare:stock_zh_a_spot_em",
                symbol="600519",
                params={"symbol": "600519"},
                ts="2026-09-12T10:30:00+08:00",
                rows=1,
                cols=3,
                fingerprint="abc123",
                from_cache=False,
            )
        ],
    )

    # 该 conversation 没有活动的注册表，因此 endpoint 必须回退
    # 到持久化的 store，而不是报告一个空作用域。
    body = client.get("/v1/citations", params={"conversation_id": "c_sources"}).json()

    assert body["count"] == 1
    citation = body["citations"][0]
    assert citation["cid"] == "cit_000001"
    assert citation["symbol"] == "600519"
    assert citation["params"] == {"symbol": "600519"}
    assert citation["from_cache"] is False


def test_session_scope_is_reclaimed_when_the_session_expires(tmp_path):
    """会话过期后其执行态必须真的被释放，而不是残留在进程级注册表里。

    这正是"每个会话残留一份记忆"的泄漏点：会话状态若在注册表之外另存一份，
    注册表淘汰就够不到它。
    """
    client, _ = make_client(tmp_path)
    client.post("/v1/chat/stream", json={"message": "第一问"})
    registry = client.app.state.session_registry
    session = next(iter(registry.sessions.values()))

    # 让该会话过期，然后触发一次淘汰（新会话即可）。
    session.last_active -= 10_000
    client.post("/v1/chat/stream", json={"message": "另一问"})

    assert session.session_id not in registry.sessions, "过期会话必须被回收"


def test_citations_by_session_come_from_the_session_loop(tmp_path):
    """按 session 读引用时数据来自会话自身，而非另一份进程级副本。"""
    from finharness.data.citation import Citation

    client, _ = make_client(tmp_path)
    client.post("/v1/chat/stream", json={"message": "第一问"})
    registry = client.app.state.session_registry
    session = next(iter(registry.sessions.values()))

    session.loop.cite.restore(
        [
            Citation(
                cid="cit_000001",
                tool="get_quote",
                endpoint="akshare:stock_zh_a_spot_em",
                symbol="600519",
                params={"symbol": "600519"},
                ts="2026-09-12T10:30:00+08:00",
                rows=1,
                cols=3,
                fingerprint="abc123",
                from_cache=False,
            )
        ]
    )

    body = client.get("/v1/citations", params={"session_id": session.session_id}).json()

    assert body["count"] == 1
    assert body["citations"][0]["cid"] == "cit_000001"


def test_conversation_citation_cache_is_bounded(tmp_path):
    """对话级引用缓存不能只增不减：进程见过的对话数可以远超活跃数。"""
    from finharness.data.citation import CitationRegistry

    client, _ = make_client(tmp_path)
    cache = client.app.state.conversation_citations
    limit = cache.max_size
    assert limit > 0, "生产装配必须给出上限"

    for index in range(limit + 10):
        cache[f"c_{index}"] = CitationRegistry()

    assert len(cache) == limit, "超出上限后必须淘汰，而不是继续增长"

