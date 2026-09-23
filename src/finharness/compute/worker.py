"""无状态远程计算 worker 的拉取循环。

本模块不导入数据库、Provider 或服务设置；它只经 HMAC 内部 API 获取单个任务包。
具体计算器必须显式注册，未知 kind 一律失败，不能把任务包当作代码执行。
"""

from __future__ import annotations

import base64
import json
import os
import socket
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from finharness.compute.protocol import TaskPackageError, TaskSigner, extract_task_package


TaskHandler = Callable[[Mapping[str, Any], Path], dict[str, Any]]


def _docx_export_handler(_job: Mapping[str, Any], input_dir: Path) -> dict[str, Any]:
    """首个迁移任务：由 worker 把受控 markdown 包导出为 DOCX。"""
    try:
        request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
        markdown = request["markdown"]
        topic = request.get("topic", "")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TaskPackageError("docx 任务包缺少合法 request.json") from exc
    if not isinstance(markdown, str) or not isinstance(topic, str) or len(markdown) > 2_000_000:
        raise TaskPackageError("docx 任务输入非法或超过大小限制")
    from finharness.tools.fin.docx_export import export_markdown_to_docx

    output = input_dir.parent / "report.docx"
    export_markdown_to_docx(markdown, out_path=output, topic=topic)
    return {"blobs": {"report.docx": base64.b64encode(output.read_bytes()).decode("ascii")}}


class WorkerClient:
    """唯一的 worker 出站能力：经签名向主服务领取/完成任务。"""

    def __init__(self, *, base_url: str, secret: str, worker_id: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.worker_id = worker_id or f"worker-{socket.gethostname()}-{os.getpid()}"
        self.signer = TaskSigner(secret)
        self.http = httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))

    def request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        response = self.http.post(
            f"{self.base_url}{path}", content=body,
            headers={"content-type": "application/json", **self.signer.sign(body)},
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def run_once(self, handlers: Mapping[str, TaskHandler]) -> bool:
        leased = self.request("/v1/internal/compute/lease", {"worker_id": self.worker_id})
        job = leased.get("job")
        if not isinstance(job, dict):
            return False
        job_id = str(job["job_id"])
        base = {"worker_id": self.worker_id, "job_id": job_id}
        self.request("/v1/internal/compute/running", base)
        try:
            package = base64.b64decode(str(leased["package_b64"]), validate=True)
            handler = handlers.get(str(job["kind"]))
            if handler is None:
                raise TaskPackageError(f"worker 不支持任务类型：{job['kind']}")
            with tempfile.TemporaryDirectory(prefix="finh-worker-") as temporary:
                task_dir = Path(temporary)
                archive = task_dir / "input.zip"
                archive.write_bytes(package)
                input_dir = task_dir / "input"
                extract_task_package(archive, input_dir)
                result = handler(job, input_dir)
            self.request("/v1/internal/compute/finish", {**base, "result_json": json.dumps(result)})
        except Exception as exc:  # noqa: BLE001 - failure is reported to main service, then loop continues.
            self.request("/v1/internal/compute/finish", {**base, "error": f"worker_error:{type(exc).__name__}"})
        return True

    def close(self) -> None:
        self.http.close()


def main() -> None:
    """worker 进程入口；业务镜像通过环境变量注入唯一的内部凭据。"""
    base_url = os.environ["FINH_COMPUTE_APP_URL"]
    secret = os.environ["FINH_COMPUTE_HMAC_SECRET"]
    poll_seconds = max(float(os.getenv("FINH_COMPUTE_POLL_SECONDS", "1")), 0.1)
    client = WorkerClient(base_url=base_url, secret=secret)
    try:
        while True:
            if not client.run_once({"docx_export": _docx_export_handler}):
                time.sleep(poll_seconds)
    finally:
        client.close()


if __name__ == "__main__":  # pragma: no cover - 容器入口
    main()
