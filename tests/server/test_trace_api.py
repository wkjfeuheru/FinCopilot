"""监控查询 API：admin 白名单鉴权与端点形状（docs 03.14.4）。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.observability.trace_store import TraceStore
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app

from tests.server.conftest import register_and_login


def _settings(tmp_path, *, enabled: bool = True, admin_users: list[str] | None = None) -> Settings:
    payload = {
        "model": {"provider": "fake"},
        "observability": {
            "trace_store": {
                "enabled": enabled,
                "db_path": str(tmp_path / "trace.db"),
                "admin_users": admin_users or [],
            }
        },
        "paths": {
            "state_dir": str(tmp_path / "state"),
            "memory_db": str(tmp_path / "state" / "m.db"),
            "auth_db": str(tmp_path / "state" / "a.db"),
            "config_db": str(tmp_path / "state" / "c.db"),
        },
    }
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return Settings.from_file(config_path)


def test_disabled_store_reports_503_with_hint(tmp_path):
    """未启用时数据端点返回 503（带开启方法），而不是无信息量的 404。"""
    client = TestClient(
        create_app(
            provider=FakeProvider(["hi"]),
            settings=_settings(tmp_path, enabled=False, admin_users=["admin"]),
        )
    )
    _, headers = register_and_login(client, "admin")
    response = client.get("/v1/trace/runs", headers=headers)
    assert response.status_code == 503
    # detail 必须可操作：告诉运维方如何开启。
    assert "trace_store.enabled" in response.json()["detail"]
    # status 端点始终可用，供前端区分"未启用"与"无权限"。
    status = client.get("/v1/trace/status", headers=headers)
    assert status.status_code == 200
    assert status.json() == {"enabled": False, "is_admin": True, "admin_configured": True}


def test_admin_filter_by_status_excludes_running(tmp_path):
    """运行中的行不计入完成率分子，且 status 过滤可用。"""
    settings = _settings(tmp_path, admin_users=["admin"])
    client = TestClient(create_app(provider=FakeProvider(["hi"]), settings=settings))
    _, headers = register_and_login(client, "admin")
    store = TraceStore(settings.observability.trace_store.db_path)
    store.start_run(run_id="tr_done", source="eval", input="a")
    store.finish_run("tr_done", status="done", succeeded=True)
    store.start_run(run_id="tr_running", source="eval", input="b")  # 无终态

    all_runs = client.get("/v1/trace/runs", headers=headers, params={"status": "done"}).json()
    assert all_runs["total"] == 1
    assert all_runs["runs"][0]["run_id"] == "tr_done"
    metrics = client.get("/v1/trace/metrics", headers=headers).json()
    assert metrics["total_runs"] == 2
    assert metrics["completion"]["task_completion_rate"] == 0.5


def test_admin_can_read_runs_and_metrics(tmp_path):
    settings = _settings(tmp_path, admin_users=["admin"])
    client = TestClient(create_app(provider=FakeProvider(["hi"]), settings=settings))
    _, headers = register_and_login(client, "admin")

    # 预置一次运行，避免依赖真实对话链路的时序。
    store = TraceStore(settings.observability.trace_store.db_path)
    store.start_run(run_id="tr_seed", source="server", user_id="u", input="茅台股价")
    store.record_event("tr_seed", "tool_status", {"name": "get_quote", "status": "started", "call_id": "a"})
    store.finish_run("tr_seed", status="done", succeeded=True, reason="done", rounds=1, tool_calls=1)

    runs = client.get("/v1/trace/runs", headers=headers)
    assert runs.status_code == 200
    body = runs.json()
    assert body["total"] == 1
    assert body["runs"][0]["run_id"] == "tr_seed"

    detail = client.get("/v1/trace/runs/tr_seed", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["status"] == "done"
    assert "rounds_trace" in detail.json()

    metrics = client.get("/v1/trace/metrics", headers=headers)
    assert metrics.status_code == 200
    m = metrics.json()
    assert m["total_runs"] == 1
    assert m["completion"]["task_completion_rate"] == 1.0
    assert m["tool_calls_total"] == 1


def test_non_admin_gets_403(tmp_path):
    settings = _settings(tmp_path, admin_users=["admin"])
    client = TestClient(create_app(provider=FakeProvider(["hi"]), settings=settings))
    _, headers = register_and_login(client, "guest")
    assert client.get("/v1/trace/runs", headers=headers).status_code == 403
    assert client.get("/v1/trace/metrics", headers=headers).status_code == 403
    assert client.get("/v1/trace/runs/tr_seed", headers=headers).status_code == 403


def test_unauthenticated_gets_401(tmp_path):
    settings = _settings(tmp_path, admin_users=["admin"])
    client = TestClient(create_app(provider=FakeProvider(["hi"]), settings=settings))
    assert client.get("/v1/trace/runs").status_code == 401


def test_missing_run_returns_404(tmp_path):
    settings = _settings(tmp_path, admin_users=["admin"])
    client = TestClient(create_app(provider=FakeProvider(["hi"]), settings=settings))
    _, headers = register_and_login(client, "admin")
    assert client.get("/v1/trace/runs/tr_absent", headers=headers).status_code == 404
