"""删除对话：移除其作用域内的记忆，保留该用户的偏好。"""

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app
from tests.server.conftest import authed_client


def make_client(tmp_path) -> TestClient:
    settings = Settings(
        paths={"memory_db": tmp_path / "memory.db", "output_dir": tmp_path / "output"},
        data={"cache_dir": tmp_path / "cache"},
    )
    return authed_client(
        TestClient(create_app(provider=FakeProvider(["ok"]), settings=settings))
    )


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
    """结论与引用随对话一同删除，而不只是删除列表项。"""
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


def test_delete_spares_user_preferences(tmp_path):
    """偏好由该用户的所有对话共享，因此删除其中一个对话会保留它们。"""
    client = make_client(tmp_path)
    cid = conversation_id_from(client.post("/v1/chat/stream", json={"message": "对话"}).text)
    store = client.app.state.memory_store
    user_id = client.finharness_user["id"]
    store.set_note("report_style", "简洁", user_id=user_id)

    client.delete(f"/v1/conversations/{cid}")

    assert store.get_notes(user_id=user_id) == {"report_style": "简洁"}


def test_delete_removes_episodes_sourced_from_the_conversation(tmp_path):
    """删除对话时，其来源的跨对话情节连同蒸馏台账一并清理。

    否则"删了对话，记忆还在"——下次开新对话时首轮仍会被重新注入该情节。
    """
    client = make_client(tmp_path)
    cid = conversation_id_from(client.post("/v1/chat/stream", json={"message": "对话"}).text)
    store = client.app.state.memory_store
    user_id = client.finharness_user["id"]
    store.add_ltm_episode(
        kind="task_result", summary="该对话产生的结论", user_id=user_id,
        source_conversation_id=cid,
    )
    store.add_ltm_episode(
        kind="decision", summary="别的对话的情节", user_id=user_id,
        source_conversation_id="c_other",
    )

    client.delete(f"/v1/conversations/{cid}")

    remaining = store.list_ltm_episodes(user_id=user_id)
    assert [episode.summary for episode in remaining] == ["别的对话的情节"]


def test_delete_unknown_conversation_is_a_404(tmp_path):
    client = make_client(tmp_path)

    assert client.delete("/v1/conversations/c_missing").status_code == 404


def test_delete_is_refused_while_the_conversation_is_busy(tmp_path):
    """在请求进行中删除，会使正在运行的循环向已不存在的目标持久化数据。"""
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
    """删除会释放该 id；客户端若回传过期的 id，则从头开始新对话。"""
    client = make_client(tmp_path)
    cid = conversation_id_from(client.post("/v1/chat/stream", json={"message": "第一问"}).text)
    client.delete(f"/v1/conversations/{cid}")

    # 复用该 id 不应使已删除的历史复活。
    client.post("/v1/chat/stream", json={"conversation_id": cid, "message": "第二问"})

    store = client.app.state.memory_store
    assert store.get_conversation(cid) is not None
    contents = [m.content for m in store.load_messages(cid)]
    assert "第二问" in contents
    assert "第一问" not in contents, "the deleted transcript must not come back"
