import json
import base64

from fastapi.testclient import TestClient

from finharness.compute.protocol import TaskSigner
from finharness.config.settings import Settings
from finharness.server.api import create_app


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
