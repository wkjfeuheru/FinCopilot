import asyncio
import json
import os
import zipfile
import io
from pathlib import Path

import pytest

from finharness.compute.executor import (
    ComputeTask,
    LocalProcessComputeExecutor,
    RemoteComputeExecutor,
)
from finharness.compute.queue import ComputeJobStore


def child_noisy(_job, _input_dir):
    import sys
    sys.stdout.write("x" * (2 * 1024 * 1024))
    return {"metadata": {}}


def child_descendant(_job, input_dir):
    import subprocess
    import sys
    import time
    from pathlib import Path
    marker = (input_dir / "marker.txt").read_text()
    code = "import time; from pathlib import Path; time.sleep(6); Path(" + repr(marker) + ").write_text('alive')"
    subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    Path(marker + ".ready").write_text("ready")
    time.sleep(20)
    return {}


@pytest.mark.asyncio
async def test_local_stdout_has_bounded_size(tmp_path):
    executor = LocalProcessComputeExecutor(handlers={"noisy": child_noisy}, work_dir=tmp_path)
    executor.max_stdout_bytes = 1024
    result = await executor.execute(_task("noisy"))
    assert result.error == "output_limit"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.asyncio
async def test_local_timeout_or_cancel_kills_descendants(tmp_path, cancel):
    marker = tmp_path / "descendant.txt"
    executor = LocalProcessComputeExecutor(handlers={"tree": child_descendant}, work_dir=tmp_path / "work")
    task = ComputeTask("u", "c", "tree", {"marker.txt": str(marker).encode()}, timeout_s=4)
    pending = asyncio.create_task(executor.execute(task))
    for _ in range(500):
        if marker.with_suffix(".txt.ready").exists():
            break
        await asyncio.sleep(0.01)
    assert marker.with_suffix(".txt.ready").exists(), (pending.result() if pending.done() else "handler must start")
    if cancel:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        assert (await pending).error == "timeout"
    await asyncio.sleep(6.2)
    assert not marker.exists(), "后代进程在任务终止后仍在运行"


@pytest.mark.asyncio
async def test_local_rejects_oversized_compressed_package_before_spawning(tmp_path):
    executor = LocalProcessComputeExecutor(handlers={"pid": child_pid}, work_dir=tmp_path)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("request.json", b"{}")
    package = archive.getvalue() + b"x" * (16 * 1024 * 1024)
    with pytest.raises(ValueError, match="大小|体积"):
        await executor.execute(ComputeTask("u", "c", "pid", {}, package_bytes=package))


def test_package_builder_rejects_compressed_size_over_limit():
    from finharness.compute.executor import _package
    with pytest.raises(ValueError, match="大小|体积"):
        _package({"input.bin": os.urandom(16 * 1024 * 1024)})


def child_pid(_job, _input_dir):
    return {"metadata": {"pid": os.getpid()}, "blobs": {}}


def child_input_path(_job, input_dir):
    return {"metadata": {"input_path": str(input_dir)}, "blobs": {}}


@pytest.mark.asyncio
async def test_extracted_inputs_stay_in_parent_owned_cleanup_directory(tmp_path):
    executor = LocalProcessComputeExecutor(handlers={"path": child_input_path}, work_dir=tmp_path)
    result = await executor.execute(_task("path", timeout_s=5))
    assert result.status == "succeeded"
    path = Path(result.metadata["input_path"])
    assert path.is_relative_to(tmp_path)
    assert not path.exists()


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
    executor = RemoteComputeExecutor(store=store, packages_dir=root, output_dir=tmp_path / "output", poll_interval_s=0.01)
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
    # 产物由主服务按 (user, conversation, job) 落盘；executor 只负责还原路径，
    # 不信任 worker 自报的任何路径。
    expected = tmp_path / "output" / "u_1" / "c_1" / leased.job_id / "a.txt"
    assert result.artifact_paths == (str(expected),)
    assert not packages[0].exists()


@pytest.mark.asyncio
async def test_remote_executor_cancellation_updates_queue_and_cleans_package(tmp_path):
    root = tmp_path / "packages"
    store = ComputeJobStore(tmp_path / "jobs.db")
    executor = RemoteComputeExecutor(store=store, packages_dir=root, output_dir=tmp_path / "output", poll_interval_s=0.01)
    pending = asyncio.create_task(executor.execute(_task()))
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    row = store._connect().execute("SELECT status FROM compute_jobs").fetchone()
    assert row["status"] == "cancelled"
    assert list(root.iterdir()) == []
