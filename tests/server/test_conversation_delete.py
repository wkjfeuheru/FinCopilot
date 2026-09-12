"""Deleting a conversation: removes its scoped memory, spares global state."""

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app


def make_client(tmp_path) -> TestClient:
    settings = Settings(
        paths={"memory_db": tmp_path / "memory.db", "output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )
    return TestClient(create_app(provider=FakeProvider(["ok"]), settings=settings))


def conversation_id_from(text: str) -> str:
    return text.split('"conversation_id": "', 1)[1].split('"', 1)[0]


def test_delete_removes_the_conversation(tmp_path):
    client = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "要删除的对话"})
    cid = conversation_id_from(first.text)

    response = client.delete(f"/v1/conversations/{cid}")

    assert response.status_code == 200
    assert response.json()["conversation_id"] == cid
    assert client.get("/v1/conversations").json()["conversations"] == []
    assert client.get(f"/v1/conversations/{cid}/messages").status_code == 404


def test_delete_removes_the_scoped_memory(tmp_path):
    """Conclusions and citations go with the conversation, not just its listing."""
    client = make_client(tmp_path)
    first = client.post("/v1/chat/stream", json={"message": "对话"})
    cid = conversation_id_from(first.text)
    store = client.app.state.memory_store
    store.save_conclusion(cid, subject="600519", text="结论", cids=["cit_000001"])
    store.upsert_symbol(cid, "600519")

    client.delete(f"/v1/conversations/{cid}")

    assert store.load_conclusions(cid) == []
    assert store.load_symbols(cid) == []
    assert store.get_conversation(cid) is None


def test_delete_leaves_other_conversations_alone(tmp_path):
    client = make_client(tmp_path)
    keep = conversation_id_from(client.post("/v1/chat/stream", json={"message": "保留"}).text)
    drop = conversation_id_from(client.post("/v1/chat/stream", json={"message": "删除"}).text)

    client.delete(f"/v1/conversations/{drop}")

    remaining = [item["conversation_id"] for item in client.get("/v1/conversations").json()["conversations"]]
    assert remaining == [keep]


def test_delete_spares_global_preferences(tmp_path):
    """Preferences are shared across conversations, so deleting one keeps them."""
    client = make_client(tmp_path)
    cid = conversation_id_from(client.post("/v1/chat/stream", json={"message": "对话"}).text)
    store = client.app.state.memory_store
    store.set_note("report_style", "简洁")

    client.delete(f"/v1/conversations/{cid}")

    assert store.get_notes() == {"report_style": "简洁"}


def test_delete_unknown_conversation_is_a_404(tmp_path):
    client = make_client(tmp_path)

    assert client.delete("/v1/conversations/c_missing").status_code == 404


def test_delete_is_refused_while_the_conversation_is_busy(tmp_path):
    """Deleting mid-request would leave the running loop persisting into nothing."""
    client = make_client(tmp_path)
    cid = conversation_id_from(client.post("/v1/chat/stream", json={"message": "对话"}).text)
    registry = client.app.state.session_registry
    session = registry.find_by_conversation(cid)
    assert session is not None
    session.busy = True

    response = client.delete(f"/v1/conversations/{cid}")

    assert response.status_code == 409
    assert client.app.state.memory_store.get_conversation(cid) is not None


def test_deleted_conversation_can_be_recreated_under_the_same_id(tmp_path):
    """Deleting frees the id; a client echoing a stale id starts fresh."""
    client = make_client(tmp_path)
    cid = conversation_id_from(client.post("/v1/chat/stream", json={"message": "第一问"}).text)
    client.delete(f"/v1/conversations/{cid}")

    # Reusing the id should not resurrect the deleted history.
    client.post("/v1/chat/stream", json={"conversation_id": cid, "message": "第二问"})

    store = client.app.state.memory_store
    assert store.get_conversation(cid) is not None
    contents = [m.content for m in store.load_messages(cid)]
    assert "第二问" in contents
    assert "第一问" not in contents, "the deleted transcript must not come back"
