"""无状态远程计算 worker 的拉取循环。

本模块不导入数据库、Provider 或服务设置；它只经 HMAC 内部 API 获取单个任务包。
具体计算器必须显式注册，未知 kind 一律失败，不能把任务包当作代码执行。
"""

from __future__ import annotations

import base64
import asyncio
import contextlib
import json
import os
import re
import socket
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from finharness.compute.protocol import TaskPackageError, TaskSigner, extract_task_package
from finharness.compute.executor import ComputeTask, LocalProcessComputeExecutor


TaskHandler = Callable[[Mapping[str, Any], Path], dict[str, Any]]

# 与 docx_export 同源的图片语法；这里只负责把包内文件名解析成绝对路径。
_IMAGE_REF = re.compile(r"(!\[.*?\]\()(<[^>]+>|[^)\s]+)(\))")


class _LeaseLost(RuntimeError):
    """续租失败：主服务已取消或回收任务，worker 不得再提交结果。"""


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

    # 图表随任务包一起送达（worker 不挂载 output 目录），此处把它们解析到
    # 解包目录内的绝对路径——导出器按相对路径会以 CWD 为基准而找不到文件。
    output = input_dir.parent / "report.docx"
    export_markdown_to_docx(_absolutize_images(markdown, input_dir), out_path=output, topic=topic)
    return {"blobs": {"report.docx": base64.b64encode(output.read_bytes()).decode("ascii")}}


def _absolutize_images(markdown: str, input_dir: Path) -> str:
    """把引用包内文件的图片目标改写为绝对路径；其余引用原样保留。"""

    def replace(match: "re.Match[str]") -> str:
        prefix, target, suffix = match.group(1), match.group(2), match.group(3)
        wrapped = target.startswith("<") and target.endswith(">")
        inner = target[1:-1].replace("\\>", ">") if wrapped else target
        candidate = input_dir / inner
        if candidate.is_file():
            return f"{prefix}{candidate}{suffix}"
        return match.group(0)

    return _IMAGE_REF.sub(replace, markdown)


class WorkerClient:
    """唯一的 worker 出站能力：经签名向主服务领取/完成任务。"""

    def __init__(self, *, base_url: str, secret: str, worker_id: str | None = None,
                 renew_interval_s: float = 15.0, work_dir: str | Path | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.worker_id = worker_id or f"worker-{socket.gethostname()}-{os.getpid()}"
        self.signer = TaskSigner(secret)
        self.http = httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))
        self.renew_interval_s = max(renew_interval_s, 0.01)
        self.work_dir = work_dir

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
            if str(job["kind"]) not in handlers:
                raise TaskPackageError(f"worker 不支持任务类型：{job['kind']}")
            result = asyncio.run(self._run_child(job, package, handlers, base))
            if result is None:  # 租约被取消或过期，不能再提交迟到的结果。
                return True
            if result.status == "succeeded":
                self.request("/v1/internal/compute/finish", {
                    **base, "result_json": json.dumps({"metadata": result.metadata, "blobs": result.blob_data})
                })
            else:
                self.request("/v1/internal/compute/finish", {**base, "error": result.error or "child_failed"})
        except _LeaseLost:
            return True
        except Exception as exc:  # noqa: BLE001 - failure is reported to main service, then loop continues.
            self.request("/v1/internal/compute/finish", {**base, "error": f"worker_error:{type(exc).__name__}"})
        return True

    async def _run_child(self, job: Mapping[str, Any], package: bytes,
                         handlers: Mapping[str, TaskHandler], base: dict[str, str]):
        executor = LocalProcessComputeExecutor(handlers=handlers, work_dir=self.work_dir)
        task = ComputeTask(user_id=str(job["user_id"]), conversation_id=str(job["conversation_id"]),
                           kind=str(job["kind"]), files={}, package_bytes=package)
        running = asyncio.create_task(executor.execute(task))

        async def renew():
            while True:
                await asyncio.sleep(self.renew_interval_s)
                await asyncio.to_thread(self.request, "/v1/internal/compute/renew", base)

        heartbeat = asyncio.create_task(renew())
        try:
            done, _ = await asyncio.wait({running, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                running.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await running
                try:
                    heartbeat.result()
                except Exception as exc:  # noqa: BLE001 - 任何续租拒绝均意味着失去租约。
                    raise _LeaseLost() from exc
                raise _LeaseLost()
            return running.result()
        finally:
            heartbeat.cancel()
            # 续租任务的原始拒绝已转换为 _LeaseLost；finally 不得以同一个
            # ValueError 覆盖它，否则外层会误把取消当成普通失败并提交 finish。
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await heartbeat

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
