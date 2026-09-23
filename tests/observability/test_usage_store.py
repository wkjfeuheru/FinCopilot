"""用量账本测试：每轮一行、与 trace 开关解耦、聚合查询。

账本（usage.db）回答"每个用户花了多少 token、聊了多少轮"——管理员页的
数据源。写入纪律与 TraceStore 一致：公开方法绝不抛出，落账失败只留日志，
绝不影响正在服务的对话。
"""

from __future__ import annotations

import sqlite3

from finharness.observability.usage_store import UsageStore


def test_record_and_totals_by_user(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.db")
    store.record_turn(
        user_id="u_1",
        input_tokens=100,
        output_tokens=50,
        cache_hit_tokens=20,
        duration_ms=1500,
        status="done",
    )
    store.record_turn(
        user_id="u_1", input_tokens=10, output_tokens=5, status="stopped"
    )
    store.record_turn(user_id="u_2", input_tokens=7, output_tokens=3, status="done")

    totals = {row["user_id"]: row for row in store.totals_by_user()}
    assert totals["u_1"]["turns"] == 2
    assert totals["u_1"]["input_tokens"] == 110
    assert totals["u_1"]["output_tokens"] == 55
    assert totals["u_1"]["cache_hit_tokens"] == 20
    assert totals["u_2"]["turns"] == 1


def test_totals_with_since_window(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.db")
    store.record_turn(user_id="u_1", input_tokens=1, output_tokens=1, status="done")
    # 手动插入一条旧记录（30 天前），窗口过滤后不应出现
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "INSERT INTO usage_turns (user_id, ts, input_tokens, output_tokens,"
            " cache_hit_tokens, duration_ms, status) VALUES (?, ?, 999, 999, 0, 0, 'done')",
            ("u_1", "2020-01-01T00:00:00+00:00"),
        )
    totals = store.totals_by_user(since="2025-01-01T00:00:00+00:00")
    assert len(totals) == 1
    assert totals[0]["input_tokens"] == 1


def test_summary_aggregates_window(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.db")
    store.record_turn(user_id="u_1", input_tokens=10, output_tokens=5, status="done")
    store.record_turn(user_id="u_2", input_tokens=1, output_tokens=1, status="done")
    summary = store.summary()
    assert summary["turns"] == 2
    assert summary["active_users"] == 2
    assert summary["input_tokens"] == 11
    assert summary["output_tokens"] == 6


def test_record_failure_never_raises(tmp_path) -> None:
    """落账是旁路：表被破坏时 record_turn 只降级，绝不让对话流失败。"""
    store = UsageStore(tmp_path / "usage.db")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("DROP TABLE usage_turns")
    store.record_turn(user_id="u_1", input_tokens=1, output_tokens=1, status="done")


def test_ping(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.db")
    store.ping()
