import asyncio
import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finharness.compute.protocol import TaskSigner
from finharness.config.settings import Settings
from finharness.server.api import create_app


def _running_job(app, tmp_path):
    package = tmp_path / "state" / "compute_packages" / "job.zip"
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_bytes(b"input")
    store = app.state.compute_jobs
    job = store.enqueue(user_id="u_1", conversation_id="c_1", kind="x", payload_path=str(package))
    store.lease_next(worker_id="w", lease_seconds=30)
    store.mark_running(job.job_id, worker_id="w", lease_seconds=30)
    return job, package


def test_cancel_racing_finish_leaves_no_output(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    job, package = _running_job(app, tmp_path)
    succeed = app.state.compute_jobs.succeed
    def cancel_before_finish(*args, **kwargs):
        app.state.compute_jobs.cancel(job.job_id, user_id="u_1")
        return succeed(*args, **kwargs)
    monkeypatch.setattr(app.state.compute_jobs, "succeed", cancel_before_finish)
    body = json.dumps({"worker_id": "w", "job_id": job.job_id,
                       "result_json": '{"blobs":{"out.txt":"eA=="}}'}).encode()
    response = TestClient(app).post("/v1/internal/compute/finish", content=body, headers=_headers(body))
    assert response.status_code == 409
    assert not list((tmp_path / "output").rglob("out.txt"))
    assert not package.exists()


def test_lease_expiring_during_blob_write_rolls_back_and_removes_output(tmp_path, monkeypatch):
    import time
    clock = [time.time()]
    monkeypatch.setattr("finharness.compute.queue.time.time", lambda: clock[0])
    app = _app(tmp_path, monkeypatch)
    job, package = _running_job(app, tmp_path)
    write = Path.write_bytes
    def expire_after_write(path, content):
        count = write(path, content)
        if path.name == "out.txt":
            clock[0] += 30
        return count
    monkeypatch.setattr(Path, "write_bytes", expire_after_write)
    body = json.dumps({"worker_id": "w", "job_id": job.job_id,
                       "result_json": '{"blobs":{"out.txt":"eA=="}}'}).encode()
    response = TestClient(app).post("/v1/internal/compute/finish", content=body, headers=_headers(body))
    assert response.status_code == 409
    assert not list((tmp_path / "output").rglob("out.txt"))
    assert app.state.compute_jobs.get(job.job_id, user_id="u_1").status == "running"
    assert package.exists(), "重试仍需要输入包"


def test_invalid_result_racing_cancel_returns_conflict_and_cleans_package(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    job, package = _running_job(app, tmp_path)
    fail = app.state.compute_jobs.fail
    def cancel_before_fail(*args, **kwargs):
        app.state.compute_jobs.cancel(job.job_id, user_id="u_1")
        return fail(*args, **kwargs)
    monkeypatch.setattr(app.state.compute_jobs, "fail", cancel_before_fail)
    body = json.dumps({"worker_id": "w", "job_id": job.job_id,
                       "result_json": '{"blobs":{"../escape":"eA=="}}'}).encode()
    response = TestClient(app).post("/v1/internal/compute/finish", content=body, headers=_headers(body))
    assert response.status_code == 409
    assert not package.exists()


def test_expired_terminal_jobs_are_cleaned_without_live_submitter(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    job, package = _running_job(app, tmp_path)
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"keep")
    other = app.state.compute_jobs.enqueue(user_id="other", conversation_id="c", kind="x", payload_path=str(outside))
    with app.state.compute_jobs._connect() as conn:
        conn.execute("UPDATE compute_jobs SET attempts=99, lease_expires_at=0 WHERE job_id=?", (job.job_id,))
    app.state.compute_jobs.cancel(other.job_id, user_id="other")
    body = b'{"worker_id":"next"}'
    response = TestClient(app).post("/v1/internal/compute/lease", content=body, headers=_headers(body))
    assert response.status_code == 200
    assert app.state.compute_jobs.get(job.job_id, user_id="u_1").status == "failed"
    assert not package.exists()
    assert outside.read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_session_dispatch_uses_settings_package_path_and_session_identity(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    assert hasattr(app.state, "compute_executor"), "主服务必须实例化远程 executor"
    session = await app.state.session_registry.ensure(user_id="u_1", conversation_id="c_1")
    pending = asyncio.create_task(session.compute.execute(kind="x", files={"input.txt": b"x"}, timeout_s=5))
    for _ in range(100):
        job = app.state.compute_jobs.lease_next(worker_id="w", lease_seconds=30)
        if job:
            break
        await asyncio.sleep(0.01)
    assert job.user_id == "u_1" and job.conversation_id == "c_1"
    assert str(tmp_path / "state" / "compute_packages") in job.payload_path
    app.state.compute_jobs.mark_running(job.job_id, worker_id="w", lease_seconds=30)
    app.state.compute_jobs.succeed(job.job_id, worker_id="w", result_json='{"metadata":{"ok":true}}')
    assert (await pending).metadata == {"ok": True}


@pytest.mark.asyncio
async def test_loop_receives_a_session_bound_compute_channel(tmp_path, monkeypatch):
    """隔离通道必须在建会话时注入 loop，且身份是会话的 user/conversation。"""
    app = _app(tmp_path, monkeypatch)
    loop = app.state.session_registry.loop_factory("s_1", "c_1", "u_1")

    assert loop.compute is not None
    assert loop.compute.user_id == "u_1"
    assert loop.compute.conversation_id == "c_1"


@pytest.mark.asyncio
async def test_no_compute_channel_without_a_configured_worker(tmp_path, monkeypatch):
    """未配 worker 的本地模式不得启用隔离通道——否则 docx 会被拖到超时。"""
    settings = Settings(
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
    )
    app = create_app(settings=settings)

    assert app.state.compute_executor is None
    loop = app.state.session_registry.loop_factory("s_1", "c_1", "u_1")
    assert loop.compute is None


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("FINH_COMPUTE_HMAC_SECRET", "test-secret-at-least-16-bytes")
    settings = Settings(
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
            "compute_jobs_db": tmp_path / "state" / "compute_jobs.db",
            "compute_packages_dir": tmp_path / "state" / "compute_packages",
        },
        # 配了远程 worker 地址，隔离计算通道才启用（未配置则工具进程内执行）。
        compute={"remote_worker_url": "http://worker.internal:8080"},
    )
    return create_app(settings=settings)


def _headers(body: bytes) -> dict[str, str]:
    return {"content-type": "application/json", **TaskSigner("test-secret-at-least-16-bytes").sign(body)}


def test_worker_lease_requires_hmac_and_returns_only_its_job_package(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    package = tmp_path / "state" / "compute_packages" / "job.zip"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"task package")
    app.state.compute_jobs.enqueue(
        user_id="u_1", conversation_id="c_1", kind="chart", payload_path=str(package)
    )
    client = TestClient(app)

    assert client.post("/v1/internal/compute/lease", json={"worker_id": "worker_1"}).status_code == 401

    body = json.dumps({"worker_id": "worker_1"}, separators=(",", ":")).encode()
    response = client.post("/v1/internal/compute/lease", content=body, headers=_headers(body))

    assert response.status_code == 200
    result = response.json()
    assert result["job"]["user_id"] == "u_1"
    assert result["package_b64"] == "dGFzayBwYWNrYWdl"


def test_worker_finish_materializes_validated_blobs_in_the_job_workspace(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    package = tmp_path / "state" / "compute_packages" / "job.zip"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"task package")
    job = app.state.compute_jobs.enqueue(
        user_id="u_1", conversation_id="c_1", kind="docx_export", payload_path=str(package)
    )
    client = TestClient(app)
    lease_body = b'{"worker_id":"worker_1"}'
    client.post("/v1/internal/compute/lease", content=lease_body, headers=_headers(lease_body))
    running_body = json.dumps({"worker_id": "worker_1", "job_id": job.job_id}, separators=(",", ":")).encode()
    assert client.post("/v1/internal/compute/running", content=running_body, headers=_headers(running_body)).status_code == 204
    finish = {
        "worker_id": "worker_1", "job_id": job.job_id,
        "result_json": json.dumps({"blobs": {"report.docx": base64.b64encode(b"docx").decode()}}),
    }
    finish_body = json.dumps(finish, separators=(",", ":")).encode()
    assert client.post("/v1/internal/compute/finish", content=finish_body, headers=_headers(finish_body)).status_code == 204
    assert (tmp_path / "output" / "u_1" / "c_1" / job.job_id / "report.docx").read_bytes() == b"docx"
    assert not package.exists()


def test_invalid_worker_result_fails_job_and_cleans_input_package(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    package = tmp_path / "state" / "compute_packages" / "job.zip"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"task package")
    job = app.state.compute_jobs.enqueue(
        user_id="u_1", conversation_id="c_1", kind="chart", payload_path=str(package)
    )
    client = TestClient(app)
    lease_body = b'{"worker_id":"worker_1"}'
    client.post("/v1/internal/compute/lease", content=lease_body, headers=_headers(lease_body))
    running_body = json.dumps({"worker_id": "worker_1", "job_id": job.job_id}, separators=(",", ":")).encode()
    client.post("/v1/internal/compute/running", content=running_body, headers=_headers(running_body))
    finish_body = json.dumps({"worker_id": "worker_1", "job_id": job.job_id,
                              "result_json": '{"blobs":{"../escape":"eA=="}}'},
                             separators=(",", ":")).encode()
    response = client.post("/v1/internal/compute/finish", content=finish_body, headers=_headers(finish_body))
    assert response.status_code == 422
    assert app.state.compute_jobs.get(job.job_id, user_id="u_1").status == "failed"
    assert not package.exists()
