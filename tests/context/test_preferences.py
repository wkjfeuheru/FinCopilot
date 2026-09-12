"""Global preferences: the one memory surface shared by every conversation."""

import asyncio

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.context.memory.store import MemoryStore
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry
from finharness.server.api import create_app
from finharness.tools.meta.preference import RememberPreferenceTool


def test_preference_is_written_to_the_global_notes_table(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    settings = Settings(paths={"memory_db": tmp_path / "memory.db"}, data={"cache_dir": tmp_path / "cache"})
    ctx = ResearchContext(cite=CitationRegistry(), settings=settings)
    ctx.store = store
    data = DataAccess([], settings=settings)

    result = asyncio.run(
        RememberPreferenceTool(data, ctx=ctx).run(key="report_style", value="简洁")
    )

    assert result.ok is True
    assert "对所有对话生效" in result.content
    # Written globally, so another conversation sees it.
    assert store.get_notes() == {"report_style": "简洁"}
    assert ctx.notes["report_style"] == "简洁"


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
        paths={"memory_db": tmp_path / "memory.db"},
        data={"cache_dir": tmp_path / "cache"},
    )
    app = create_app(settings=settings)
    store: MemoryStore = app.state.memory_store
    store.ensure_conversation("c_1", title="茅台分析")
    store.set_note("report_style", "简洁")
    store.save_conclusion("c_1", subject="600519", text="ROE 30%", cids=["cit_000001"])

    client = TestClient(app)
    body = client.get("/v1/memory", params={"conversation_id": "c_1"}).json()

    assert body["notes"] == {"report_style": "简洁"}
    assert any(item["conversation_id"] == "c_1" for item in body["conversations"])
    assert body["conclusions"][0]["text"] == "ROE 30%"
    assert body["conclusions"][0]["subject"] == "600519"


def test_memory_endpoint_omits_conclusions_without_a_conversation(tmp_path):
    settings = Settings(
        paths={"memory_db": tmp_path / "memory.db"},
        data={"cache_dir": tmp_path / "cache"},
    )
    app = create_app(settings=settings)
    app.state.memory_store.set_note("k", "v")

    body = TestClient(app).get("/v1/memory").json()

    assert body["notes"] == {"k": "v"}
    assert body["conclusions"] == []
