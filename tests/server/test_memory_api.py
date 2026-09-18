"""跨对话长期记忆的 API 治理面（docs 03.6.4）：查看 / 编辑 / 删除。"""

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.server.api import create_app
from tests.server.conftest import authed_client


def make_client(tmp_path) -> TestClient:
    return authed_client(
        TestClient(
            create_app(
                settings=Settings(
                    paths={
                        "memory_db": tmp_path / "memory.db",
                        "auth_db": tmp_path / "users.db",
                    },
                    data={"cache_dir": tmp_path / "cache"},
                )
            )
        )
    )


def test_memory_view_lists_episodes_with_provenance(tmp_path):
    client = make_client(tmp_path)
    user_id = client.finharness_user["id"]
    store = client.app.state.memory_store
    store.add_ltm_episode(
        kind="task_result",
        summary="茅台毛利率 91%",
        user_id=user_id,
        subject="600519",
        cids=["cit_000001"],
        source_conversation_id="c_a",
        source_title="茅台分析",
        source_ts="2026-01-02T00:00:00+00:00",
    )

    body = client.get("/v1/memory").json()

    assert len(body["episodes"]) == 1
    episode = body["episodes"][0]
    assert episode["summary"] == "茅台毛利率 91%"
    assert episode["source_title"] == "茅台分析"
    assert episode["cids"] == ["cit_000001"]
    assert episode["ep_uid"].startswith("ep_")


def test_memory_view_filters_episodes_by_subject_and_kind(tmp_path):
    client = make_client(tmp_path)
    user_id = client.finharness_user["id"]
    store = client.app.state.memory_store
    store.add_ltm_episode(kind="task_result", summary="a", user_id=user_id, subject="600519")
    store.add_ltm_episode(kind="decision", summary="b", user_id=user_id, subject="000858")

    assert len(client.get("/v1/memory", params={"subject": "600519"}).json()["episodes"]) == 1
    assert len(client.get("/v1/memory", params={"kind": "decision"}).json()["episodes"]) == 1
    assert len(client.get("/v1/memory").json()["episodes"]) == 2


def test_episode_patch_updates_content(tmp_path):
    client = make_client(tmp_path)
    user_id = client.finharness_user["id"]
    store = client.app.state.memory_store
    record = store.add_ltm_episode(kind="decision", summary="旧内容", user_id=user_id)

    response = client.patch(
        f"/v1/memory/episodes/{record.ep_uid}", json={"summary": "新内容", "subject": "600519"}
    )

    assert response.status_code == 200
    updated = store.get_ltm_episode(record.ep_uid, user_id=user_id)
    assert updated.summary == "新内容"
    assert updated.subject == "600519"


def test_episode_patch_rejects_unknown_or_foreign_episode(tmp_path):
    client = make_client(tmp_path)
    store = client.app.state.memory_store
    record = store.add_ltm_episode(kind="decision", summary="别人的", user_id="someone_else")

    assert client.patch(
        "/v1/memory/episodes/ep_doesnotexist", json={"summary": "x"}
    ).status_code == 404
    # 归属不符视同不存在。
    assert client.patch(
        f"/v1/memory/episodes/{record.ep_uid}", json={"summary": "x"}
    ).status_code == 404
    assert store.get_ltm_episode(record.ep_uid, user_id="someone_else").summary == "别人的"


def test_episode_delete_removes_it(tmp_path):
    client = make_client(tmp_path)
    user_id = client.finharness_user["id"]
    store = client.app.state.memory_store
    record = store.add_ltm_episode(kind="excerpt", summary="片段", user_id=user_id)

    response = client.delete(f"/v1/memory/episodes/{record.ep_uid}")

    assert response.status_code == 200
    assert store.get_ltm_episode(record.ep_uid, user_id=user_id) is None
    assert client.delete(f"/v1/memory/episodes/{record.ep_uid}").status_code == 404


# --- 语义记忆（facts）--------------------------------------------------------

def test_memory_view_lists_facts_and_reports_semantic_search(tmp_path):
    client = make_client(tmp_path)
    user_id = client.finharness_user["id"]
    store = client.app.state.memory_store
    store.upsert_ltm_fact(
        user_id=user_id, key="report_style", statement="报告要简洁",
        kind="preference",
    )
    store.upsert_ltm_fact(
        user_id=user_id, key="maotai", statement="茅台属于白酒行业",
        kind="fact", subject="600519",
    )

    body = client.get("/v1/memory").json()

    keys = {item["key"] for item in body["facts"]}
    assert keys == {"report_style", "maotai"}
    # 偏好同时以 notes 形状返回，兼容既有前端与调用方。
    assert body["notes"] == {"report_style": "报告要简洁"}
    # 未配 embedding 端点时如实报告语义召回不可用。
    assert body["semantic_search"] is False
    assert all(item["fa_uid"].startswith("fa_") for item in body["facts"])


def test_memory_view_filters_facts_by_kind(tmp_path):
    client = make_client(tmp_path)
    user_id = client.finharness_user["id"]
    store = client.app.state.memory_store
    store.upsert_ltm_fact(user_id=user_id, key="a", statement="x", kind="fact")
    store.upsert_ltm_fact(user_id=user_id, key="b", statement="y", kind="concept")

    body = client.get("/v1/memory", params={"kind": "concept"}).json()

    assert [item["key"] for item in body["facts"]] == ["b"]


def test_fact_patch_overrides_the_statement(tmp_path):
    client = make_client(tmp_path)
    user_id = client.finharness_user["id"]
    store = client.app.state.memory_store
    fact = store.upsert_ltm_fact(
        user_id=user_id, key="report_style", statement="用表格", kind="preference"
    )

    response = client.patch(
        f"/v1/memory/facts/{fact.fa_uid}", json={"statement": "简洁，少用表格"}
    )

    assert response.status_code == 200
    assert store.get_notes(user_id=user_id) == {"report_style": "简洁，少用表格"}


def test_fact_patch_and_delete_are_user_scoped(tmp_path):
    client = make_client(tmp_path)
    store = client.app.state.memory_store
    fact = store.upsert_ltm_fact(user_id="someone_else", key="k", statement="别人的")

    assert client.patch(
        f"/v1/memory/facts/{fact.fa_uid}", json={"statement": "x"}
    ).status_code == 404
    assert client.delete(f"/v1/memory/facts/{fact.fa_uid}").status_code == 404
    assert store.get_ltm_fact(fact.fa_uid, user_id="someone_else").statement == "别人的"


def test_fact_delete_removes_it(tmp_path):
    client = make_client(tmp_path)
    user_id = client.finharness_user["id"]
    store = client.app.state.memory_store
    fact = store.upsert_ltm_fact(user_id=user_id, key="k", statement="v")

    response = client.delete(f"/v1/memory/facts/{fact.fa_uid}")

    assert response.status_code == 200
    assert store.get_ltm_fact(fact.fa_uid, user_id=user_id) is None
    assert client.delete(f"/v1/memory/facts/{fact.fa_uid}").status_code == 404
