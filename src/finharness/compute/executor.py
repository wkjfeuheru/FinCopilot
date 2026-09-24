"""计算任务的两个执行适配器；调用方只需等待 ``execute`` 的结构化结果。"""

from __future__ import annotations

import asyncio
import base64
import io
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from finharness.compute.queue import ComputeJobStore
from finharness.compute.protocol import MAX_PACKAGE_BYTES, MAX_PACKAGE_FILES
from finharness.compute.process_tree import ProcessTree


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ComputeTask:
    user_id: str
    conversation_id: str
    kind: str
    files: Mapping[str, bytes]
    timeout_s: float = 120.0
    package_bytes: bytes | None = None


@dataclass(frozen=True, slots=True)
class ComputeResult:
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    blobs: tuple[str, ...] = ()
    error: str | None = None
    job_id: str | None = None
    blob_data: Mapping[str, str] = field(default_factory=dict)


class ComputeExecutor(Protocol):
    async def execute(self, job: ComputeTask) -> ComputeResult: ...


def _package(files: Mapping[str, bytes]) -> bytes:
    if not files or len(files) > MAX_PACKAGE_FILES:
        raise ValueError("任务包文件数量非法")
    if sum(len(content) for content in files.values()) > MAX_PACKAGE_BYTES:
        raise ValueError("任务包超过大小限制")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in files.items():
            if not isinstance(name, str) or not _NAME.fullmatch(name) or not isinstance(content, bytes):
                raise ValueError("任务包文件名或内容非法")
            bundle.writestr(name, content)
    package = output.getvalue()
    if len(package) > MAX_PACKAGE_BYTES:
        raise ValueError("任务压缩包超过大小限制")
    return package


def _result(payload: object, *, job_id: str | None = None) -> ComputeResult:
    if not isinstance(payload, dict):
        raise ValueError("结果必须是 object")
    metadata = payload.get("metadata", {})
    blobs = payload.get("blobs", {})
    if not isinstance(metadata, dict) or not isinstance(blobs, dict) or len(blobs) > 16:
        raise ValueError("结果元数据或 blob 非法")
    for name, encoded in blobs.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name) or not isinstance(encoded, str):
            raise ValueError("blob 名称或内容非法")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("blob 必须是 base64") from exc
        if len(decoded) > 32 * 1024 * 1024:
            raise ValueError("blob 超过大小限制")
    return ComputeResult("succeeded", metadata, tuple(blobs), job_id=job_id, blob_data=blobs)


class RemoteComputeExecutor:
    """主服务适配器：生成任务包、排队、异步轮询并清理输入。"""

    def __init__(self, *, store: ComputeJobStore, packages_dir: str | Path,
                 poll_interval_s: float = 0.1) -> None:
        self.store = store
        self.packages_dir = Path(packages_dir).resolve()
        self.poll_interval_s = max(poll_interval_s, 0.001)

    def cleanup_terminal_packages(self) -> None:
        for job in self.store.recover_and_list_terminal():
            path = Path(job.payload_path).resolve()
            if path.is_relative_to(self.packages_dir) and path.is_file():
                path.unlink(missing_ok=True)

    async def execute(self, job: ComputeTask) -> ComputeResult:
        if job.timeout_s <= 0:
            raise ValueError("任务超时必须为正数")
        package = _package(job.files)
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        path = self.packages_dir / f"job_{uuid.uuid4().hex}.zip"
        path.write_bytes(package)
        queued = None
        try:
            queued = await asyncio.to_thread(
                self.store.enqueue, user_id=job.user_id, conversation_id=job.conversation_id,
                kind=job.kind, payload_path=str(path), job_id=path.stem,
            )
            deadline = asyncio.get_running_loop().time() + job.timeout_s
            while True:
                current = await asyncio.to_thread(self.store.get, queued.job_id, user_id=job.user_id)
                if current is None:
                    return ComputeResult("failed", error="job_missing", job_id=queued.job_id)
                if current.status == "succeeded":
                    try:
                        payload = json.loads(current.result_json or "{}")
                        metadata = payload.get("metadata", {})
                        artifacts = payload.get("artifacts", [])
                        if not isinstance(metadata, dict) or not isinstance(artifacts, list) or any(
                            not isinstance(name, str) or not _NAME.fullmatch(name) for name in artifacts
                        ):
                            raise ValueError("远程结果非法")
                    except (json.JSONDecodeError, AttributeError, ValueError):
                        return ComputeResult("failed", error="invalid_result", job_id=queued.job_id)
                    return ComputeResult("succeeded", metadata, tuple(artifacts), job_id=queued.job_id)
                if current.status in {"failed", "cancelled"}:
                    return ComputeResult(current.status, error=current.error, job_id=queued.job_id)
                if asyncio.get_running_loop().time() >= deadline:
                    await asyncio.to_thread(self.store.cancel, queued.job_id, user_id=job.user_id)
                    return ComputeResult("failed", error="timeout", job_id=queued.job_id)
                await asyncio.sleep(self.poll_interval_s)
        except asyncio.CancelledError:
            if queued is not None:
                await asyncio.to_thread(self.store.cancel, queued.job_id, user_id=job.user_id)
            raise
        finally:
            path.unlink(missing_ok=True)


class LocalProcessComputeExecutor:
    """仅供本地单租户使用，显式注册的 handler 在新进程中执行。"""

    def __init__(self, *, handlers: Mapping[str, Callable], work_dir: str | Path | None = None,
                 cpu_seconds: int = 30, memory_bytes: int = 512 * 1024 * 1024,
                 max_stdout_bytes: int = 48 * 1024 * 1024) -> None:
        self.handlers = dict(handlers)
        self.work_dir = Path(work_dir).resolve() if work_dir is not None else None
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_bytes
        if max_stdout_bytes <= 0:
            raise ValueError("输出大小限制必须为正数")
        self.max_stdout_bytes = max_stdout_bytes

    async def execute(self, job: ComputeTask) -> ComputeResult:
        if job.timeout_s <= 0:
            raise ValueError("任务超时必须为正数")
        handler = self.handlers.get(job.kind)
        if handler is None or "<locals>" in handler.__qualname__:
            return ComputeResult("failed", error="unsupported_kind")
        package = job.package_bytes if job.package_bytes is not None else _package(job.files)
        if len(package) > MAX_PACKAGE_BYTES:
            raise ValueError("任务压缩包超过大小限制")
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="finh-compute-", dir=self.work_dir))
        child = None
        tree = None
        try:
            archive = temporary / "input.zip"
            archive.write_bytes(package)
            preexec = _linux_limits(temporary, self.cpu_seconds, self.memory_bytes)
            env = _minimal_env(handler)
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "finharness.compute.child", str(archive),
                f"{handler.__module__}:{handler.__qualname__}", job.kind, str(temporary),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                env=env, preexec_fn=preexec,
                start_new_session=os.name != "nt",
            )
            tree = ProcessTree(child.pid)
            child.stdin.write(b"1")
            await child.stdin.drain()
            child.stdin.close()

            async def read_result() -> bytes:
                output = bytearray()
                while chunk := await child.stdout.read(min(65536, self.max_stdout_bytes + 1 - len(output))):
                    output.extend(chunk)
                    if len(output) > self.max_stdout_bytes:
                        raise OverflowError("output_limit")
                await child.wait()
                return bytes(output)

            try:
                stdout = await asyncio.wait_for(read_result(), timeout=job.timeout_s)
            except TimeoutError:
                return ComputeResult("failed", error="timeout")
            except OverflowError:
                return ComputeResult("failed", error="output_limit")
            if child.returncode != 0:
                return ComputeResult("failed", error="child_failed")
            try:
                return _result(json.loads(stdout), job_id=None)
            except (ValueError, json.JSONDecodeError):
                return ComputeResult("failed", error="invalid_result")
        finally:
            if tree is not None:
                tree.terminate()
            elif child is not None and child.returncode is None:
                child.kill()
            if child is not None:
                # 输出超限可能暂停 StreamReader；终止后只排空有界管道缓冲，
                # 否则 asyncio 的进程退出通知会一直等待管道关闭。
                while await child.stdout.read(65536):
                    pass
                await child.wait()
            shutil.rmtree(temporary, ignore_errors=True)


def _minimal_env(handler: Callable) -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[2]
    handler_root = Path(inspect.getfile(handler)).resolve().parent
    env = {"PYTHONPATH": os.pathsep.join((str(source_root), str(handler_root))),
           "PYTHONIOENCODING": "utf-8"}
    for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _linux_limits(directory: Path, cpu_seconds: int, memory_bytes: int):
    if not sys.platform.startswith("linux"):
        return None
    import pwd
    import resource

    uid_entry = pwd.getpwnam("nobody") if os.geteuid() == 0 else None
    if uid_entry is not None:
        os.chown(directory, uid_entry.pw_uid, uid_entry.pw_gid)
        os.chown(directory / "input.zip", uid_entry.pw_uid, uid_entry.pw_gid)

    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        if uid_entry is not None:
            os.setgroups([])
            os.setgid(uid_entry.pw_gid)
            os.setuid(uid_entry.pw_uid)
    return limit
