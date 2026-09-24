import base64
import io
import json
import time
import zipfile

from finharness.compute.worker import WorkerClient


def slow_handler(_job, _input_dir):
    time.sleep(0.25)
    return {"metadata": {"ok": True}, "blobs": {"answer.txt": base64.b64encode(b"ready").decode()}}


def _bundle():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("request.json", "{}")
    return base64.b64encode(stream.getvalue()).decode()


def test_worker_renews_lease_while_child_is_running_and_returns_named_blob(tmp_path):
    client = WorkerClient(base_url="http://unused", secret="test-secret-at-least-16-bytes",
                          renew_interval_s=0.05, work_dir=tmp_path)
    calls = []

    def request(path, payload):
        calls.append((path, payload))
        if path.endswith("/lease"):
            return {"job": {"job_id": "job_" + "a" * 32, "kind": "slow", "user_id": "u_1",
                            "conversation_id": "c_1"}, "package_b64": _bundle()}
        return {}

    client.request = request
    try:
        assert client.run_once({"slow": slow_handler}) is True
    finally:
        client.close()
    paths = [path for path, _ in calls]
    assert "/v1/internal/compute/renew" in paths
    finish = calls[-1][1]
    assert json.loads(finish["result_json"])["blobs"]["answer.txt"] == "cmVhZHk="
    assert list(tmp_path.iterdir()) == []


def test_worker_stops_child_when_renewal_reports_cancelled(tmp_path):
    client = WorkerClient(base_url="http://unused", secret="test-secret-at-least-16-bytes",
                          renew_interval_s=0.05, work_dir=tmp_path)
    calls = []

    def request(path, payload):
        calls.append(path)
        if path.endswith("/lease"):
            return {"job": {"job_id": "job_" + "a" * 32, "kind": "slow", "user_id": "u_1",
                            "conversation_id": "c_1"}, "package_b64": _bundle()}
        if path.endswith("/renew"):
            raise ValueError("任务已取消")
        return {}

    client.request = request
    try:
        assert client.run_once({"slow": slow_handler}) is True
    finally:
        client.close()
    assert "/v1/internal/compute/finish" not in calls
    assert list(tmp_path.iterdir()) == []
