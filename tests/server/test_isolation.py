"""多用户隔离：本需求的核心验收（docs 03.13）。

每个字节的用户数据——对话、消息、引用、结论、偏好、供应商配置、
产物文件——都必须对一个用户不可见，且不可被其改写。
跨用户访问一律 404（不泄露存在性），产物路径一律 403。
"""

from pathlib import Path

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.server.api import create_app
from tests.server.conftest import register_and_login


class EchoProvider:
    """离线 provider：回答会回显累计的轮次，便于确认记忆是否被隔离。"""

    def __init__(self):
        self.seen_user_turns: list[int] = []

    async def stream(self, *, system, messages, tools, usage):
        from finharness.types import ModelUsage, StreamChunk, StreamEvent

        turns = sum(1 for message in messages if message.role == "user")
        self.seen_user_turns.append(turns)
        yield StreamChunk(StreamEvent.TEXT_DELTA, f"第{turns}轮")
        yield StreamChunk(
            StreamEvent.MESSAGE_END,
            ModelUsage(input_tokens=1, output_tokens=1),
        )


def make_app(tmp_path, provider=None):
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "cache" / "memory.db",
            "auth_db": tmp_path / "cache" / "users.db",
        },
    )
    return create_app(provider or EchoProvider(), settings=settings)


def make_two_users(tmp_path):
    """构建一个共享应用，返回 (client_a, headers_a, client_b, headers_b)。

    两个客户端指向同一个 ASGI 应用（同一份共享存储），
    仅凭令牌区分身份——这正是隔离必须成立的场景。注册的身份挂在
    ``client.finharness_user`` 上供直接访问 store 的断言使用。
    """
    app = make_app(tmp_path)
    client_a = TestClient(app)
    client_b = TestClient(app)
    token_a, headers_a = register_and_login(client_a, "alice")
    token_b, headers_b = register_and_login(client_b, "bob")
    store = app.state.user_store
    client_a.finharness_user = {"id": store.find_by_username("alice").id, "username": "alice"}
    client_b.finharness_user = {"id": store.find_by_username("bob").id, "username": "bob"}
    return client_a, headers_a, client_b, headers_b


def conversation_id_from(text: str) -> str:
    return text.split('"conversation_id": "', 1)[1].split('"', 1)[0]


# -- 对话清单 / 消息 -----------------------------------------------------------

def test_conversation_lists_are_per_user(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    client_a.post("/v1/chat/stream", json={"message": "A 的对话"}, headers=headers_a)

    a_list = client_a.get("/v1/conversations", headers=headers_a).json()["conversations"]
    b_list = client_b.get("/v1/conversations", headers=headers_b).json()["conversations"]

    assert len(a_list) == 1
    assert b_list == [], "B 不能看到 A 的对话"


def test_messages_of_another_user_are_a_404(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    first = client_a.post("/v1/chat/stream", json={"message": "A 的秘密"}, headers=headers_a)
    cid = conversation_id_from(first.text)

    response = client_b.get(f"/v1/conversations/{cid}/messages", headers=headers_b)

    assert response.status_code == 404
    assert "A 的秘密" not in response.text


def test_another_user_cannot_delete_a_conversation(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    first = client_a.post("/v1/chat/stream", json={"message": "A 的对话"}, headers=headers_a)
    cid = conversation_id_from(first.text)

    assert client_b.delete(f"/v1/conversations/{cid}", headers=headers_b).status_code == 404
    # A 的对话毫发无损。
    assert client_a.get(f"/v1/conversations/{cid}/messages", headers=headers_a).status_code == 200


def test_another_user_cannot_continue_a_conversation(tmp_path):
    """核心约束：B 带着 A 的 conversation_id 提问，不得读到 A 的历史。"""
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    first = client_a.post("/v1/chat/stream", json={"message": "A 的第一问"}, headers=headers_a)
    cid = conversation_id_from(first.text)

    response = client_b.post(
        "/v1/chat/stream", json={"conversation_id": cid, "message": "B 的追问"}, headers=headers_b
    )

    assert response.status_code == 404


def test_another_user_cannot_hijack_a_session_id(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    first = client_a.post("/v1/chat/stream", json={"message": "A 的对话"}, headers=headers_a)
    session_id = first.text.split('"session_id": "', 1)[1].split('"', 1)[0]

    response = client_b.post(
        "/v1/chat/stream", json={"session_id": session_id, "message": "劫持"}, headers=headers_b
    )

    # 会话归属不符时视同不存在（409 会话正忙 / 404 都算拒绝，绝不放行）。
    assert response.status_code in (404, 409)


# -- 引用 / 结论 / 偏好 --------------------------------------------------------

def test_citations_are_not_readable_across_users(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    first = client_a.post("/v1/chat/stream", json={"message": "A 的对话"}, headers=headers_a)
    cid = conversation_id_from(first.text)

    response = client_b.get("/v1/citations", params={"conversation_id": cid}, headers=headers_b)

    assert response.status_code == 404


def test_notes_are_per_user(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    store = client_a.app.state.memory_store
    user_a = client_a.finharness_user["id"]
    user_b = client_b.finharness_user["id"]
    store.set_note("report_style", "A 的偏好", user_id=user_a)

    a_notes = client_a.get("/v1/memory", headers=headers_a).json()["notes"]
    b_notes = client_b.get("/v1/memory", headers=headers_b).json()["notes"]

    assert a_notes == {"report_style": "A 的偏好"}
    assert b_notes == {}, "偏好必须按用户隔离，绝不共享"


def test_conclusions_are_not_readable_across_users(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    first = client_a.post("/v1/chat/stream", json={"message": "A 的对话"}, headers=headers_a)
    cid = conversation_id_from(first.text)
    store = client_a.app.state.memory_store
    store.save_conclusion(cid, subject="600519", text="A 的结论", cids=[])

    body = client_b.get(
        "/v1/memory", params={"conversation_id": cid}, headers=headers_b
    ).json()

    assert body["conclusions"] == []


# -- 交互式确认 ----------------------------------------------------------------

def test_another_user_cannot_answer_a_pending_confirm(tmp_path):
    """B 拿到 A 的 request_id 也不能替 A 批准写操作。"""
    app = make_app(tmp_path)
    client_a = TestClient(app)
    client_b = TestClient(app)
    register_and_login(client_a, "alice")
    _, headers_b = register_and_login(client_b, "bob")
    bus = app.state.confirm_bus

    import asyncio

    async def scenario():
        async def answer():
            await asyncio.sleep(0.01)
            pending = bus.pending_ids()
            assert pending, "request should be registered"
            # B 尝试代答 A 的请求。
            denied = client_b.post(
                "/v1/chat/respond",
                json={"request_id": pending[-1], "response": "y"},
                headers=headers_b,
            )
            assert denied.status_code == 404
            return pending[-1]

        task = asyncio.create_task(answer())
        # 该请求归属 A（user_id="u_a"）；B 的令牌对应另一个 id。
        _, result = await bus.request(
            session_id="s", kind="confirm", prompt="?", user_id="u_a", ttl_s=0.5
        )
        await task
        return result

    # 超时收场：B 的应答必须没有生效。
    assert asyncio.run(scenario()) is None


# -- 供应商配置 ----------------------------------------------------------------

def test_provider_configs_are_per_user(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    payload = {
        "name": "deepseek",
        "kind": "fake",
        "model": "fake-model",
        "api_key": None,
        "activate": True,
    }
    created = client_a.post("/v1/config", json=payload, headers=headers_a)
    assert created.status_code == 200
    config_id = created.json()["config"]["id"]

    b_view = client_b.get("/v1/config", headers=headers_b).json()

    assert b_view["configs"] == []
    assert b_view["configured"] is False
    # B 也不能按 id 操作 A 的配置。
    assert client_b.put(
        f"/v1/config/{config_id}", json=payload, headers=headers_b
    ).status_code == 404
    assert client_b.post(
        f"/v1/config/{config_id}/activate", headers=headers_b
    ).status_code == 404
    assert client_b.delete(f"/v1/config/{config_id}", headers=headers_b).status_code == 404


def test_same_config_name_is_allowed_for_different_users(tmp_path):
    client_a, headers_a, client_b, headers_b = make_two_users(tmp_path)
    payload = {"name": "shared-name", "kind": "fake", "model": "m", "activate": True}

    first = client_a.post("/v1/config", json=payload, headers=headers_a)
    second = client_b.post("/v1/config", json=payload, headers=headers_b)

    assert first.status_code == 200
    assert second.status_code == 200, "名称唯一性按用户计算"


# -- 产物文件 ------------------------------------------------------------------

def test_artifacts_are_scoped_to_the_owner(tmp_path):
    app = make_app(tmp_path)
    client = TestClient(app)
    store = app.state.user_store
    alice = store.register("alice", "secret-pass-1").user
    bob = store.register("bob", "secret-pass-1").user
    headers_a = {"Authorization": f"Bearer {store.login('alice', 'secret-pass-1').token}"}
    headers_b = {"Authorization": f"Bearer {store.login('bob', 'secret-pass-1').token}"}

    output = tmp_path / "output"
    (output / alice.id).mkdir(parents=True, exist_ok=True)
    (output / bob.id).mkdir(parents=True, exist_ok=True)
    (output / alice.id / "report.md").write_text("# A 的报告", encoding="utf-8")
    (output / bob.id / "report.md").write_text("# B 的报告", encoding="utf-8")

    a_response = client.get(
        "/v1/artifacts", params={"path": str(output / alice.id / "report.md")}, headers=headers_a
    )
    b_crossing = client.get(
        "/v1/artifacts", params={"path": str(output / alice.id / "report.md")}, headers=headers_b
    )

    assert a_response.status_code == 200
    assert "A 的报告" in a_response.text
    assert b_crossing.status_code == 403, "B 不能下载 A 的产物"


def test_artifact_endpoint_does_not_serve_the_user_database(tmp_path):
    """data_cache 只开放 parquet/ 子树，否则 users.db 可被直接下载。"""
    client_a, headers_a, _, _ = make_two_users(tmp_path)
    users_db = Path(client_a.app.state.user_store.db_path)

    response = client_a.get(
        "/v1/artifacts", params={"path": str(users_db)}, headers=headers_a
    )

    assert response.status_code == 403


def test_artifact_endpoint_does_not_serve_the_memory_database(tmp_path):
    client_a, headers_a, _, _ = make_two_users(tmp_path)
    memory_db = Path(client_a.app.state.memory_store.db_path)

    response = client_a.get(
        "/v1/artifacts", params={"path": str(memory_db)}, headers=headers_a
    )

    assert response.status_code == 403


# -- 记忆不被跨用户汇聚 ---------------------------------------------------------

def test_a_turn_does_not_see_another_users_history(tmp_path):
    """A 在同一对话内续问应看到 2 轮；B 即使另起对话，其首轮也必须是第 1 轮。"""
    provider = EchoProvider()
    app = make_app(tmp_path, provider=provider)
    client_a = TestClient(app)
    client_b = TestClient(app)
    _, headers_a = register_and_login(client_a, "alice")
    _, headers_b = register_and_login(client_b, "bob")

    first = client_a.post("/v1/chat/stream", json={"message": "A 问一"}, headers=headers_a)
    cid = conversation_id_from(first.text)
    # 续问必须带 conversation_id，否则每次提问都会开一个新对话。
    client_a.post(
        "/v1/chat/stream",
        json={"message": "A 问二", "conversation_id": cid},
        headers=headers_a,
    )
    client_b.post("/v1/chat/stream", json={"message": "B 的第一问"}, headers=headers_b)

    assert provider.seen_user_turns == [1, 2, 1], "B 的记忆不得从 A 的对话续接"
