"""用户偏好：该用户所有对话共享的唯一 memory 面（doc 03.13）。"""

import asyncio

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.context.memory.store import MemoryStore
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.server.api import create_app
from finharness.tools.meta.preference import RememberPreferenceTool
from tests.server.conftest import authed_client


def test_preference_is_written_to_the_users_notes_table(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = Settings(paths={"memory_db": tmp_path / "memory.db"}, data={"cache_dir": tmp_path / "cache"})
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    ctx.store = store
    ctx.user_id = "u_1"
    data = DataAccess([], settings=settings)

    result = asyncio.run(
        RememberPreferenceTool(data, ctx=ctx).run(key="report_style", value="简洁")
    )

    assert result.ok is True
    assert "对所有对话生效" in result.content
    # 按用户写入，因此该用户的其他对话也能看到它。
    assert store.get_notes(user_id="u_1") == {"report_style": "简洁"}
    assert ctx.notes["report_style"] == "简洁"
    # 其他用户看不到。
    assert store.get_notes(user_id="u_2") == {}


def test_preference_tool_is_read_only_so_it_needs_no_confirmation(tmp_path):
    from finharness.tools.base import PermissionLevel

    assert RememberPreferenceTool.permission is PermissionLevel.READ


def test_preference_without_a_store_still_updates_the_session(tmp_path):
    settings = Settings(data={"cache_dir": tmp_path / "cache"})
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    data = DataAccess([], settings=settings)

    result = asyncio.run(
        RememberPreferenceTool(data, ctx=ctx).run(key="style", value="简洁")
    )

    assert result.ok is True
    assert "仅当前会话有效" in result.content
    assert ctx.notes["style"] == "简洁"


def test_memory_endpoint_reports_notes_and_conversations(tmp_path):
    settings = Settings(
        paths={
            "memory_db": tmp_path / "memory.db",
            "auth_db": tmp_path / "users.db",
        },
        data={"cache_dir": tmp_path / "cache"},
    )
    client = authed_client(TestClient(create_app(settings=settings)))
    user_id = client.finharness_user["id"]
    store: MemoryStore = client.app.state.memory_store
    store.ensure_conversation("c_1", user_id=user_id, title="茅台分析")
    store.set_note("report_style", "简洁", user_id=user_id)
    store.save_conclusion("c_1", subject="600519", text="ROE 30%", cids=["cit_000001"])

    body = client.get("/v1/memory", params={"conversation_id": "c_1"}).json()

    assert body["notes"] == {"report_style": "简洁"}
    assert any(item["conversation_id"] == "c_1" for item in body["conversations"])
    assert body["conclusions"][0]["text"] == "ROE 30%"
    assert body["conclusions"][0]["subject"] == "600519"


def test_memory_endpoint_omits_conclusions_without_a_conversation(tmp_path):
    settings = Settings(
        paths={
            "memory_db": tmp_path / "memory.db",
            "auth_db": tmp_path / "users.db",
        },
        data={"cache_dir": tmp_path / "cache"},
    )
    client = authed_client(TestClient(create_app(settings=settings)))
    client.app.state.memory_store.set_note(
        "k", "v", user_id=client.finharness_user["id"]
    )

    body = client.get("/v1/memory").json()

    assert body["notes"] == {"k": "v"}
    assert body["conclusions"] == []
