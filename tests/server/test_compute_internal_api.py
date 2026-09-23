import json

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
