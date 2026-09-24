"""治理事件的指标化：拒绝、配额、审计失败要能告警，而不只是写日志。

隔离方案 Phase 2 的告警前提——此前这些事件只出现在日志里，没有可聚合的
计数器，"拒绝率突然上升"就无法成为一条告警规则。测试从 HTTP 面触发一次
真实的配额拒绝，断言它落进了 ``governance_events_total``。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from finharness.config.settings import Settings
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app
from tests.server.conftest import authed_client


def _metrics_client(tmp_path, **quota) -> TestClient:
    settings = Settings(
        model={"provider": "fake"},
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "out",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
        quota=quota,
        observability={"metrics": {"enabled": True}},
    )
    return authed_client(
        TestClient(create_app(FakeProvider(["hi", " there"]), settings=settings))
    )


def test_turn_budget_rejection_lands_in_the_governance_counter(tmp_path):
    """轮次预算拒绝 → 429，且 governance_events_total 记为 quota_turn_budget。"""
    client = _metrics_client(tmp_path, turns_per_window=1, window_s=3600)

    def turn() -> int:
        with client.stream("POST", "/v1/chat/stream", json={"message": "hi"}) as response:
            for _ in response.iter_lines():
                pass
            return response.status_code

    assert turn() == 200  # 第一轮用掉预算
    assert turn() == 429  # 第二轮被拒
    body = client.get("/metrics").text
    assert 'governance_events_total{kind="quota_turn_budget"}' in body
