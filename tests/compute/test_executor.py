import asyncio
import json
import os
import zipfile

import pytest

from finharness.compute.executor import (
    ComputeTask,
    LocalProcessComputeExecutor,
    RemoteComputeExecutor,
)
from finharness.compute.queue import ComputeJobStore


def child_pid(_job, _input_dir):
    return {"metadata": {"pid": os.getpid()}, "blobs": {}}


def child_sleep(_job, _input_dir):
    import time

    time.sleep(5)
    return {"metadata": {}, "blobs": {}}


def child_invalid(_job, _input_dir):
    return {"blobs": {"../escape": "eA=="}}


def _task(kind="pid", timeout_s=2):
    return ComputeTask(user_id="u_1", conversation_id="c_1", kind=kind,
                       files={"request.json": b"{}"}, timeout_s=timeout_s)


@pytest.mark.asyncio
async def test_local_executor_runs_in_new_process_and_returns_structured_result(tmp_path):
    executor = LocalProcessComputeExecutor(handlers={"pid": child_pid}, work_dir=tmp_path)
    result = await executor.execute(_task())
    assert result.status == "succeeded"
    assert result.metadata == {"pid": result.metadata["pid"]}
    assert result.metadata["pid"] != os.getpid()


@pytest.mark.asyncio
async def test_local_executor_enforces_wall_timeout_and_cleans_package(tmp_path):
    executor = LocalProcessComputeExecutor(handlers={"sleep": child_sleep}, work_dir=tmp_path)
    result = await executor.execute(_task("sleep", timeout_s=0.1))
    assert result.status == "failed"
    assert result.error == "timeout"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_local_executor_rejects_invalid_named_blob(tmp_path):
    executor = LocalProcessComputeExecutor(handlers={"invalid": child_invalid}, work_dir=tmp_path)
    result = await executor.execute(_task("invalid"))
    assert result.status == "failed"
    assert result.error == "invalid_result"


@pytest.mark.asyncio
async def test_local_executor_cancellation_kills_child_and_cleans_package(tmp_path):
    executor = LocalProcessComputeExecutor(handlers={"sleep": child_sleep}, work_dir=tmp_path)
    pending = asyncio.create_task(executor.execute(_task("sleep")))
    await asyncio.sleep(0.15)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_remote_executor_writes_only_to_package_root_and_waits_without_blocking(tmp_path):
    root = tmp_path / "state" / "packages"
    store = ComputeJobStore(tmp_path / "state" / "jobs.db")
    executor = RemoteComputeExecutor(store=store, packages_dir=root, poll_interval_s=0.01)
    pending = asyncio.create_task(executor.execute(_task()))
    await asyncio.sleep(0.05)
    assert not pending.done()
    packages = list(root.glob("*.zip"))
    assert len(packages) == 1
    with zipfile.ZipFile(packages[0]) as bundle:
        assert bundle.read("request.json") == b"{}"
    jobs = store._connect().execute("SELECT job_id, payload_path FROM compute_jobs").fetchall()
    assert len(jobs) == 1
    assert packages[0].stem == jobs[0]["job_id"]
    leased = store.lease_next(worker_id="w", lease_seconds=10)
    assert leased is not None
    store.mark_running(leased.job_id, worker_id="w", lease_seconds=10)
    store.succeed(leased.job_id, worker_id="w", result_json=json.dumps({"metadata": {"ok": True}, "artifacts": ["a.txt"]}))
    result = await pending
    assert result.status == "succeeded"
    assert result.metadata == {"ok": True}
    assert result.blobs == ("a.txt",)
    assert not packages[0].exists()


@pytest.mark.asyncio
async def test_remote_executor_cancellation_updates_queue_and_cleans_package(tmp_path):
    root = tmp_path / "packages"
    store = ComputeJobStore(tmp_path / "jobs.db")
    executor = RemoteComputeExecutor(store=store, packages_dir=root, poll_interval_s=0.01)
    pending = asyncio.create_task(executor.execute(_task()))
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    row = store._connect().execute("SELECT status FROM compute_jobs").fetchone()
    assert row["status"] == "cancelled"
    assert list(root.iterdir()) == []
