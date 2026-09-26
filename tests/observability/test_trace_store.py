"""TraceStore：落库、指标口径、脱敏与"落库失败不影响主流程"（docs 03.14.4）。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from finharness.observability.trace_store import TraceStore


def _round(turn: int, *, thought: str = "", answer: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        turn=turn,
        thought=thought,
        actions=[SimpleNamespace(call_id=f"c{turn}", name="get_quote", args={"symbol": "600519"})],
        observations=[
            SimpleNamespace(
                call_id=f"c{turn}", name="get_quote", ok=True, error=None,
                preview="1700.5", duration_ms=120,
            )
        ],
        input_tokens=10,
        output_tokens=5,
        llm_first_ms=200,
        llm_ms=900,
        answer=answer,
    )


@pytest.fixture()
def store(tmp_path):
    return TraceStore(tmp_path / "trace.db")


def test_run_roundtrip_persists_trace_and_rounds(store):
    store.start_run(run_id="tr_1", user_id="u1", conversation_id="conv1", input="茅台多少钱？")
    store.record_event("tr_1", "tool_status", {"name": "get_quote", "status": "started"})
    store.finish_run(
        "tr_1", status="done", answer="1700 元", reason="done", succeeded=True,
        rounds=1, tool_calls=1, usage={"input_tokens": 10, "output_tokens": 5},
        citations=["cid_1"], trace_rounds=[_round(1, thought="需要先取价")],
    )
    detail = store.run_detail("tr_1")
    assert detail is not None
    assert detail["status"] == "done"
    assert detail["succeeded"] == 1
    assert detail["citations"] == ["cid_1"]
    assert detail["rounds_trace"][0]["thought"] == "需要先取价"
    # 参数以脱敏文本快照存储，而非原始 dict。
    assert detail["rounds_trace"][0]["actions"][0]["args"] == "symbol=600519"
    assert detail["rounds_trace"][0]["observations"][0]["preview"] == "1700.5"
    assert [e["kind"] for e in detail["events"]] == ["tool_status"]


def test_metrics_separate_blocks_from_failures(store):
    store.start_run(run_id="r1", input="写文件")
    # 成功调用
    store.record_event("r1", "tool_status", {"name": "get_quote", "status": "started", "call_id": "a"})
    store.record_event("r1", "tool_status", {"name": "get_quote", "status": "completed", "ok": True, "call_id": "a"})
    # 权限门拒绝：没有 started 帧，带 verdict
    store.record_event("r1", "tool_status", {"name": "write_file", "status": "failed", "ok": False, "verdict": "denied", "call_id": "b"})
    # 超时
    store.record_event("r1", "tool_status", {"name": "get_kline", "status": "started", "call_id": "c"})
    store.record_event("r1", "tool_status", {"name": "get_kline", "status": "failed", "ok": False, "verdict": "timeout", "call_id": "c"})
    store.finish_run("r1", status="done", succeeded=True, rounds=2, tool_calls=3)

    m = store.metrics_summary()
    assert m["tool_calls_total"] == 2          # 只有两个 started
    assert m["safety_blocks"] == 1             # write_file 被拒
    assert m["timeout_rate"] == 0.5
    # 被拒的 write_file 不计入失败率分子
    assert m["per_tool"]["write_file"]["blocked"] == 1
    assert m["per_tool"]["write_file"]["total"] == 0


def test_completion_rate_and_reason_breakdown(store):
    store.start_run(run_id="ok", input="a")
    store.finish_run("ok", status="done", succeeded=True, reason="done", rounds=2)
    store.start_run(run_id="bad", input="b")
    store.finish_run("bad", status="error", succeeded=False, reason="loop_detected", rounds=5)

    m = store.metrics_summary()
    assert m["total_runs"] == 2
    assert m["completion"]["task_completion_rate"] == 0.5
    assert m["completion"]["reason_counts"] == {"done": 1, "loop_detected": 1}
    assert m["avg_rounds"] == 3.5


def test_repeat_call_rate_from_loop_guard(store):
    store.start_run(run_id="r2", input="反复查")
    store.record_event("r2", "tool_status", {"name": "get_quote", "status": "started", "call_id": "a"})
    store.record_event("r2", "loop_guard", {"name": "get_quote", "count": 3, "turn": 4})
    store.record_event("r2", "loop_guard", {"name": "get_quote", "count": 4, "turn": 5})
    store.finish_run("r2", status="done", succeeded=True)

    m = store.metrics_summary()
    assert m["loop_guard_events"] == 2
    assert m["repeat_call_rate"] == 2.0  # 2 次拦截 / 1 次调用


def test_filters_scope_runs(store):
    store.start_run(run_id="s1", source="server", user_id="alice", input="x")
    store.finish_run("s1", status="done", succeeded=True)
    store.start_run(run_id="e1", source="eval", eval_case_id="NRM-001", input="y")
    store.finish_run("e1", status="done", succeeded=True)

    assert store.count_runs() == 2
    assert store.count_runs(source="eval") == 1
    assert store.count_runs(user_id="alice") == 1
    assert store.count_runs(user_id="nobody") == 0
    eval_run = store.list_runs(source="eval")[0]
    assert eval_run["eval_case_id"] == "NRM-001"


def test_secrets_are_redacted_in_args(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    store.start_run(run_id="r", input="x")
    store.finish_run(
        "r", status="done", succeeded=True,
        trace_rounds=[
            SimpleNamespace(
                turn=1, thought="", answer="",
                actions=[SimpleNamespace(call_id="c", name="web_search", args={"api_key": "sk-secret-value", "q": "茅台"})],
                observations=[], input_tokens=0, output_tokens=0, llm_first_ms=0, llm_ms=0,
            )
        ],
    )
    detail = store.run_detail("r")
    args = detail["rounds_trace"][0]["actions"][0]["args"]
    assert "sk-secret-value" not in args


def test_capture_disabled_truncates_payload(tmp_path):
    store = TraceStore(tmp_path / "t.db", capture_payloads=False)
    store.start_run(run_id="r", input="长输入" * 500)
    store.finish_run(
        "r", status="done", succeeded=True,
        trace_rounds=[_round(1, thought="想" * 5000)],
    )
    detail = store.run_detail("r")
    assert len(detail["input"]) <= 2000
    assert len(detail["rounds_trace"][0]["thought"]) <= 2000


def test_write_failure_never_raises(tmp_path):
    """库文件不可写（路径被目录占用）时，所有写入静默降级。"""
    store = TraceStore(tmp_path / "t.db")
    store.db_path = tmp_path / "t.db" / "nested" / "impossible.db"  # 无效路径
    # 这些调用都不应抛出。
    store.start_run(run_id="x", input="a")
    store.record_event("x", "tool_status", {"name": "t"})
    store.finish_run("x", status="done", succeeded=True, trace_rounds=[_round(1)])
    assert store.run_detail("x") is None or isinstance(store.run_detail("x"), dict)


def test_cleanup_removes_old_runs(store):
    store.start_run(run_id="old", input="a")
    store.finish_run("old", status="done", succeeded=True)
    # 保留 0 天 → 不清理；保留负值也不清理。
    assert store.cleanup(0) == 0
    # 强制过期：直接改 started_at。
    with store._connect() as conn:
        conn.execute("UPDATE trace_runs SET started_at = '2000-01-01T00:00:00.000+00:00' WHERE run_id='old'")
    assert store.cleanup(30) == 1
    assert store.run_detail("old") is None


def test_concurrent_record_event_assigns_unique_monotonic_seq(store):
    """同 run 的并发 record_event 必须各得唯一且连续的 seq。

    ``_seq`` 是每 run 的无锁内存序号；服务端可能从工作线程写 trace，
    因此 get/set 由锁保护、序号不可重复。
    """
    store.start_run(run_id="r", input="a")
    workers = 40
    barrier = threading.Barrier(workers)

    def record(n: int) -> None:
        barrier.wait()
        store.record_event("r", "tool_status", {"n": n})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(record, range(workers)))

    detail = store.run_detail("r")
    seqs = sorted(event["seq"] for event in detail["events"])
    assert seqs == list(range(1, workers + 1))


def test_trace_store_records_state_events_without_special_casing(store):
    """TraceStore already records every non-delta event — including FSM ``state``."""
    store.start_run(run_id="tr_state", input="resume me")
    store.record_event(
        "tr_state",
        "state",
        {"run_id": "r1", "revision": 2, "phase": "hydrate"},
    )
    detail = store.run_detail("tr_state")
    assert detail is not None
    assert [event["kind"] for event in detail["events"]] == ["state"]
    assert detail["events"][0]["payload"]["phase"] == "hydrate"


def test_state_events_also_land_in_trace_states(store):
    """``state`` 事件同时提升为 trace_states 的一等行（每步 agent 状态）。"""
    store.start_run(run_id="tr_1", input="分析茅台")
    store.record_event(
        "tr_1", "state",
        {"run_id": "agent-7", "revision": 0, "phase": "hydrate", "turn": 0,
         "created_at": "2026-09-25T00:00:00.000+00:00"},
    )
    store.record_event(
        "tr_1", "state",
        {"run_id": "agent-7", "revision": 1, "phase": "thinking", "turn": 0,
         "created_at": "2026-09-25T00:00:01.000+00:00"},
    )
    detail = store.run_detail("tr_1")
    states = detail["states"]
    assert [s["phase"] for s in states] == ["hydrate", "thinking"]
    assert [s["revision"] for s in states] == [0, 1]
    # agent_run_id 保留 FSM 运行身份，供 resume 的各段串回。
    assert states[0]["agent_run_id"] == "agent-7"
    assert states[0]["turn"] == 0
    assert states[1]["payload"]["phase"] == "thinking"


def test_state_rows_are_idempotent_on_run_revision(store):
    """同一 (run_id, revision) 重复到达（重放/重试）只保留最新一行。"""
    store.start_run(run_id="tr_1", input="x")
    store.record_event("tr_1", "state", {"run_id": "a", "revision": 3, "phase": "thinking"})
    store.record_event("tr_1", "state", {"run_id": "a", "revision": 3, "phase": "tooluse"})
    detail = store.run_detail("tr_1")
    assert len(detail["states"]) == 1
    assert detail["states"][0]["phase"] == "tooluse"


def test_record_round_is_live_and_idempotent(store):
    """实时落轮次；与 finish_run 的批量写就同一 (run_id, turn) 幂等共存。"""
    store.start_run(run_id="tr_1", input="x")
    store.record_round("tr_1", _round(1, thought="第一轮草稿"), phase="tooluse", revision=2)
    # finish 批量写同一轮（phase/revision 为 None）不得覆盖实时写入的它们。
    store.finish_run("tr_1", status="done", succeeded=True, trace_rounds=[_round(1, thought="最终")])
    detail = store.run_detail("tr_1")
    assert len(detail["rounds_trace"]) == 1
    assert detail["rounds_trace"][0]["thought"] == "最终"
    assert detail["rounds_trace"][0]["phase"] == "tooluse"
    assert detail["rounds_trace"][0]["revision"] == 2


def test_record_round_survives_without_finish(store):
    """崩溃场景：只有实时轮次、没有 finish_run 时轮次仍在库里。"""
    store.start_run(run_id="tr_crash", input="x")
    store.record_round("tr_crash", _round(1), phase="thinking", revision=1)
    detail = store.run_detail("tr_crash")
    assert len(detail["rounds_trace"]) == 1
    assert detail["status"] == "running"


def test_cleanup_removes_states_too(store):
    store.start_run(run_id="old", input="a")
    store.record_event("old", "state", {"run_id": "a", "revision": 1, "phase": "thinking"})
    store.record_round("old", _round(1))
    store.finish_run("old", status="done", succeeded=True)
    with store._connect() as conn:
        conn.execute("UPDATE trace_runs SET started_at = '2000-01-01T00:00:00.000+00:00' WHERE run_id='old'")
    assert store.cleanup(30) == 1
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM trace_states").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM trace_rounds").fetchone()["n"] == 0


def test_migration_adds_round_columns_to_legacy_db(tmp_path):
    """旧库（trace_rounds 无 phase/revision、无唯一索引）可幂等升级。"""
    import sqlite3

    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute(
        "CREATE TABLE trace_rounds (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,"
        " turn INTEGER NOT NULL, thought TEXT NOT NULL DEFAULT '',"
        " actions_json TEXT NOT NULL DEFAULT '[]', observations_json TEXT NOT NULL DEFAULT '[]',"
        " input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,"
        " llm_first_ms INTEGER NOT NULL DEFAULT 0, llm_ms INTEGER NOT NULL DEFAULT 0,"
        " answer TEXT NOT NULL DEFAULT '')"
    )
    # 历史重复行：唯一索引建立前必须被收敛。
    conn.executemany(
        "INSERT INTO trace_rounds (run_id, turn) VALUES (?, ?)",
        [("r", 1), ("r", 1), ("r", 2)],
    )
    conn.commit()
    conn.close()

    store = TraceStore(legacy)  # 迁移在 _init_db 内完成
    with store._connect() as connection:
        cols = {row["name"] for row in connection.execute("PRAGMA table_info(trace_rounds)")}
        assert {"phase", "revision"} <= cols
        # 重复的 (r, 1) 已收敛为一行。
        assert connection.execute(
            "SELECT COUNT(*) AS n FROM trace_rounds WHERE run_id='r' AND turn=1"
        ).fetchone()["n"] == 1
    # 升级后可安全实时写。
    store.record_round("r", _round(3), phase="complete", revision=5)
    detail = store.run_detail("r")
    assert detail is None or any(r["turn"] == 3 for r in detail["rounds_trace"])


def test_metrics_include_step_counts(store):
    """指标新增步数口径：总步数与平均步数（只计有状态记录的运行）。"""
    store.start_run(run_id="r1", input="a")
    for revision, phase in enumerate(("hydrate", "thinking", "complete")):
        store.record_event("r1", "state", {"run_id": "a", "revision": revision, "phase": phase})
    store.finish_run("r1", status="done", succeeded=True)

    m = store.metrics_summary()
    assert m["total_states"] == 3
    assert m["avg_steps"] == 3.0
