"""断点的存储层：写入、覆盖、读取与级联清理（docs 03.3）。

断点按 ``conversation_id`` 主键，因此"最新一行"就是恢复所需的全部状态。
这里验证的正是这条不变量的两端：覆盖写只留最新，删除对话时断点随之消失
（否则删掉的对话会一直留下一个可"继续"的幻影）。
"""

import pytest

from finharness.context.memory.store import MemoryStore
from finharness.context.session import Plan, PlanStep


def test_checkpoint_roundtrips_all_fields(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.ensure_conversation("c1", user_id="u1", title="t")

    store.save_checkpoint(
        "c1",
        status="stopped",
        reason="user_stopped",
        rounds=3,
        turn_index=3,
        plan={"plan_id": "plan_002", "goal": "研究", "revision": 2, "steps": []},
        partial_answer="部分结论",
        persisted_seq=9,
    )

    checkpoint = store.load_latest_checkpoint("c1")

    assert checkpoint is not None
    assert checkpoint.status == "stopped"
    assert checkpoint.reason == "user_stopped"
    assert checkpoint.rounds == 3
    assert checkpoint.turn_index == 3
    assert checkpoint.partial_answer == "部分结论"
    assert checkpoint.persisted_seq == 9
    assert checkpoint.plan["plan_id"] == "plan_002"
    assert checkpoint.recoverable is True


def test_latest_checkpoint_overwrites_the_previous_one(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.ensure_conversation("c1", user_id="u1", title="t")

    store.save_checkpoint("c1", status="running", rounds=1)
    store.save_checkpoint("c1", status="stopped", reason="user_stopped", rounds=5)

    checkpoint = store.load_latest_checkpoint("c1")

    assert checkpoint is not None
    # 只有最新状态留存：历史断点既无消费者也会无界增长。
    assert checkpoint.status == "stopped"
    assert checkpoint.rounds == 5


def test_running_and_completed_are_not_recoverable(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.ensure_conversation("c1", user_id="u1", title="t")

    store.save_checkpoint("c1", status="running")
    assert store.load_latest_checkpoint("c1").recoverable is False

    store.save_checkpoint("c1", status="completed")
    assert store.load_latest_checkpoint("c1").recoverable is False


def test_unknown_status_is_rejected(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.ensure_conversation("c1", user_id="u1", title="t")

    with pytest.raises(ValueError):
        store.save_checkpoint("c1", status="made-up")


def test_missing_checkpoint_reads_as_none(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    assert store.load_latest_checkpoint("never_seen") is None


def test_deleting_a_conversation_removes_its_checkpoint(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.ensure_conversation("c1", user_id="u1", title="t")
    store.save_checkpoint("c1", status="stopped", reason="user_stopped")

    store.delete_conversation("c1")

    # 否则被删对话会留下一个可"继续"的幻影。
    assert store.load_latest_checkpoint("c1") is None


def test_clear_checkpoint_removes_only_the_target(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.ensure_conversation("c1", user_id="u1", title="t")
    store.ensure_conversation("c2", user_id="u1", title="t")
    store.save_checkpoint("c1", status="stopped")
    store.save_checkpoint("c2", status="stopped")

    store.clear_checkpoint("c1")

    assert store.load_latest_checkpoint("c1") is None
    assert store.load_latest_checkpoint("c2") is not None


def test_prune_removes_checkpoints_with_their_conversations(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    for index in range(3):
        store.ensure_conversation(f"c{index}", user_id="u1", title="t")
        store.save_checkpoint(f"c{index}", status="stopped")

    store.prune(max_conversations=1, max_age_days=3650)

    remaining = [
        cid for cid in ("c0", "c1", "c2") if store.load_latest_checkpoint(cid) is not None
    ]
    assert len(remaining) <= 1
