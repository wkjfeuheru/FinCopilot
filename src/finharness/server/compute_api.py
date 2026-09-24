"""隔离计算通道的内部 worker 接口（docs 03.15）。

worker 是独立进程：它只能经这几个**签名**端点领取任务、上报进度与交回产物，绝不
直接挂载主服务的 ``state`` 或打开队列 SQLite 文件。把这段协议单独成模块，是因为
它是整个系统里唯一对"进程外、且请求可能被伪造"的一方开放的入口——它的每条校验
（HMAC 签名、任务包落在受控目录、blob 名与大小、绝不信任 worker 回传的路径）都
是安全边界，集中在一处才便于审查。

用 ``create_compute_router`` 工厂而非模块级路由，与 ``auth_api``/``config_api`` 等
一致：依赖（signer、队列、executor、settings）由 ``create_app`` 显式注入，路由本身
不持有全局状态。``state`` 即 ``application.state``——队列与 executor 的生命周期由
主服务管理，此处只读用。
"""

from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from finharness.compute.protocol import (
    MAX_PACKAGE_BYTES,
    ReplayError,
    SignatureError,
    allocate_artifact_dir,
)


class WorkerLeaseRequest(BaseModel):
    worker_id: str


class WorkerJobRequest(BaseModel):
    worker_id: str
    job_id: str


class WorkerFinishRequest(WorkerJobRequest):
    result_json: str | None = None
    error: str | None = None


def create_compute_router(*, settings: Any, state: Any) -> APIRouter:
    """构建 /v1/internal/compute/*（不进入 OpenAPI schema，非公开 API）。"""

    router = APIRouter()

    async def _verify_worker_request(request: Request) -> None:
        signer = state.compute_signer
        if signer is None:
            raise HTTPException(status_code=503, detail="计算 worker 签名密钥未配置")
        try:
            signer.verify(await request.body(), request.headers)
        except (ReplayError, SignatureError) as exc:
            raise HTTPException(status_code=401, detail="worker 请求签名无效") from exc

    def _worker_job_payload(job) -> dict:
        package_root = Path(settings.paths.compute_packages_dir).resolve()
        package_path = Path(job.payload_path).resolve()
        try:
            package_path.relative_to(package_root)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="计算任务包不在受控 state 目录") from exc
        if not package_path.is_file() or package_path.stat().st_size > MAX_PACKAGE_BYTES:
            raise HTTPException(status_code=500, detail="计算任务包不可用或超过大小限制")
        return {
            "job": {
                "job_id": job.job_id,
                "user_id": job.user_id,
                "conversation_id": job.conversation_id,
                "kind": job.kind,
                "attempts": job.attempts,
            },
            "package_b64": base64.b64encode(package_path.read_bytes()).decode("ascii"),
        }

    def _materialize_worker_blobs(job, result_json: str, created: list[Path]) -> str:
        """验证 worker 回传物后才写入所属任务目录，绝不相信其目标路径。"""
        try:
            result = json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="worker 返回的 result_json 非法") from exc
        if not isinstance(result, dict):
            raise HTTPException(status_code=422, detail="worker 结果必须是 object")
        if not isinstance(result.get("metadata", {}), dict):
            raise HTTPException(status_code=422, detail="worker 元数据必须是 object")
        blobs = result.pop("blobs", {})
        if not blobs:
            return json.dumps(result, ensure_ascii=False)
        if not isinstance(blobs, dict) or len(blobs) > 16:
            raise HTTPException(status_code=422, detail="worker blob 数量非法")
        destination = allocate_artifact_dir(
            settings.paths.output_dir,
            user_id=job.user_id,
            conversation_id=job.conversation_id,
            job_id=job.job_id,
        )
        created.append(destination)
        artifacts: list[str] = []
        try:
            for name, encoded in blobs.items():
                if not isinstance(name, str) or Path(name).name != name or not name or len(name) > 128:
                    raise HTTPException(status_code=422, detail="worker blob 名称非法")
                if not isinstance(encoded, str):
                    raise HTTPException(status_code=422, detail="worker blob 内容非法")
                try:
                    content = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise HTTPException(status_code=422, detail="worker blob 不是 base64") from exc
                if len(content) > 32 * 1024 * 1024:
                    raise HTTPException(status_code=422, detail="worker blob 超过大小限制")
                (destination / name).write_bytes(content)
                artifacts.append(name)
        except Exception:
            # 不保留失败任务的半成品，且目录只可能是本次调用刚创建的。
            import shutil

            shutil.rmtree(destination, ignore_errors=True)
            raise
        result["artifacts"] = artifacts
        return json.dumps(result, ensure_ascii=False)

    def _cleanup_compute_package(job) -> None:
        package_root = Path(settings.paths.compute_packages_dir).resolve()
        package_path = Path(job.payload_path).resolve()
        if package_path.is_relative_to(package_root):
            package_path.unlink(missing_ok=True)

    @router.post("/v1/internal/compute/lease", include_in_schema=False)
    async def lease_compute_job(payload: WorkerLeaseRequest, request: Request) -> dict:
        await _verify_worker_request(request)
        job = state.compute_jobs.lease_next(
            worker_id=payload.worker_id, lease_seconds=float(settings.compute.lease_seconds)
        )
        executor = state.compute_executor
        if executor is not None:
            executor.cleanup_terminal_packages()
        return {"job": None} if job is None else _worker_job_payload(job)

    @router.post("/v1/internal/compute/running", include_in_schema=False)
    async def mark_compute_running(payload: WorkerJobRequest, request: Request) -> Response:
        await _verify_worker_request(request)
        try:
            state.compute_jobs.mark_running(
                payload.job_id, worker_id=payload.worker_id, lease_seconds=float(settings.compute.lease_seconds)
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)

    @router.post("/v1/internal/compute/renew", include_in_schema=False)
    async def renew_compute_lease(payload: WorkerJobRequest, request: Request) -> Response:
        await _verify_worker_request(request)
        try:
            state.compute_jobs.renew(
                payload.job_id, worker_id=payload.worker_id, lease_seconds=float(settings.compute.lease_seconds)
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)

    @router.post("/v1/internal/compute/finish", include_in_schema=False)
    async def finish_compute_job(payload: WorkerFinishRequest, request: Request) -> Response:
        await _verify_worker_request(request)
        job = state.compute_jobs.get_leased(payload.job_id, worker_id=payload.worker_id)
        if job is None:
            raise HTTPException(status_code=409, detail="任务未被当前 worker 运行")
        created: list[Path] = []
        committed = False
        try:
            if payload.error:
                state.compute_jobs.fail(payload.job_id, worker_id=payload.worker_id, error=payload.error)
            else:
                state.compute_jobs.succeed(
                    payload.job_id,
                    worker_id=payload.worker_id,
                    result_json=lambda: _materialize_worker_blobs(job, payload.result_json or "{}", created),
                )
            committed = True
        except HTTPException as exc:
            if exc.status_code == 422:
                try:
                    state.compute_jobs.fail(
                        payload.job_id, worker_id=payload.worker_id, error="invalid_result"
                    )
                except ValueError as conflict:
                    raise HTTPException(status_code=409, detail=str(conflict)) from conflict
            raise
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            if not committed:
                import shutil
                for destination in created:
                    shutil.rmtree(destination, ignore_errors=True)
            current = state.compute_jobs.get(job.job_id, user_id=job.user_id)
            if current is not None and current.status in {"succeeded", "failed", "cancelled"}:
                _cleanup_compute_package(job)
        return Response(status_code=204)

    return router
