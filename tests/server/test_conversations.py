"""Resuming a conversation over HTTP: list, replay, and continue.

A session is an execution window that expires; a conversation is the memory
scope and outlives it. These tests cover the client-visible consequences: the
list endpoint, transcript replay, and that continuing by conversation id reuses
the same memory rather than starting over.
"""

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app
from finharness.types import ModelUsage, Msg, StreamChunk, StreamEvent


class EchoProvider(FakeProvider):
    """Echoes the number of prior user turns, to expose restored history."""

    def __init__(self):
        super().__init__([])
        self.seen_user_turns: list[int] = []

    async def stream(self, *, system, messages, tools, usage: ModelUsage):
        self.requests.append(list(messages))
        users = [m for m in messages if m.role == "user"]
        self.seen_user_turns.append(len(users))
        yield StreamChunk(StreamEvent.TEXT_DELTA, f"第{len(users)}问已回答")
        yield StreamChunk(
            StreamEvent.MESSAGE_END, ModelUsage(input_tokens=1, output_tokens=1)
        )


def make_client(tmp_path) -> tuple[TestClient, EchoProvider]:
    provider = EchoProvider()
    settings = Settings(
        paths={"memory_db": tmp_path / "memory.db", "output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )
    return TestClient(create_app(provider=provider, settings=settings)), provider


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


def test_replay_omits_tool_frames(tmp_path):
    """Only readable turns come back; working state is not shown to the reader."""
    client, _ = make_client(tmp_path)
    # Seed a conversation containing a tool round directly in the store.
    store = client.app.state.memory_store
    store.ensure_conversation("c_tools")
    from finharness.types import ToolUse

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
    """The core acceptance: resuming must make the model see the earlier turns."""
    client, provider = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "第一问"})
    cid = conversation_id_from(first.text)

    client.post("/v1/chat/stream", json={"conversation_id": cid, "message": "第二问"})

    # The second request should carry both user turns, not just the new one.
    assert provider.seen_user_turns[-1] == 2


def test_a_new_conversation_starts_with_no_history(tmp_path):
    client, provider = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "甲对话的问题"})
    assert first.status_code == 200

    # A request without a conversation id is a fresh conversation.
    client.post("/v1/chat/stream", json={"message": "乙对话的问题"})

    assert provider.seen_user_turns[-1] == 1
    conversations = client.get("/v1/conversations").json()["conversations"]
    assert len(conversations) == 2, "each conversation is tracked separately"


def test_resuming_after_the_session_expired_still_restores_history(tmp_path):
    """Conversation ids survive the execution window; that is why they exist."""
    client, provider = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "第一问"})
    cid = conversation_id_from(first.text)

    # Expire the live session without touching the stored conversation.
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
    assert response.json()["count"] == 0  # no tool ran, but the scope resolves


def test_citations_404_for_an_unknown_conversation(tmp_path):
    client, _ = make_client(tmp_path)

    assert (
        client.get("/v1/citations", params={"conversation_id": "c_none"}).status_code
        == 404
    )


def test_persisted_citations_are_readable_after_the_session_is_gone(tmp_path):
    """A conversation outlives the process; its sources must remain addressable."""
    from finharness.data.citation import Citation

    client, _ = make_client(tmp_path)
    store = client.app.state.memory_store
    store.ensure_conversation("c_sources")
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

    # No live registry exists for this conversation, so the endpoint must fall
    # back to the persisted store rather than reporting an empty scope.
    body = client.get("/v1/citations", params={"conversation_id": "c_sources"}).json()

    assert body["count"] == 1
    citation = body["citations"][0]
    assert citation["cid"] == "cit_000001"
    assert citation["symbol"] == "600519"
    assert citation["params"] == {"symbol": "600519"}
    assert citation["from_cache"] is False
