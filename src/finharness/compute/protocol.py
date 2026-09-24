"""主服务与无状态计算 worker 之间的最小信任边界。

任务输入永远由主服务放入 state 目录。worker 只接受带 HMAC 的请求、把包解到
自己创建的临时目录，并把命名结果交回主服务；它不需要、也不应持有租户数据库
或 ``/data`` 挂载。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import shutil
import stat
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path


class SignatureError(ValueError):
    """请求不是由可信主服务签发。"""


class ReplayError(SignatureError):
    """同一 nonce 在签名有效期内被使用了不止一次。"""


class TaskPackageError(ValueError):
    """任务压缩包不满足 worker 的安全输入约束。"""


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SAFE_JOB_ID = re.compile(r"^(?:job_)?[a-f0-9]{32}$")
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_FILES = 64


class TaskSigner:
    """为 worker 内部请求签名并在接收端防止重放。"""

    def __init__(self, secret: str | bytes, *, max_clock_skew_s: int = 60) -> None:
        encoded = secret.encode("utf-8") if isinstance(secret, str) else secret
        if len(encoded) < 16:
            raise ValueError("worker HMAC 密钥至少需要 16 字节")
        if max_clock_skew_s <= 0:
            raise ValueError("max_clock_skew_s 必须为正数")
        self._secret = encoded
        self._max_clock_skew_s = max_clock_skew_s
        self._seen_nonces: dict[str, int] = {}

    def sign(self, body: bytes, *, now: int | None = None, nonce: str | None = None) -> dict[str, str]:
        timestamp = int(time.time() if now is None else now)
        request_nonce = nonce or secrets.token_urlsafe(24)
        digest = hashlib.sha256(body).hexdigest()
        canonical = f"{timestamp}\n{request_nonce}\n{digest}".encode("ascii")
        signature = hmac.new(self._secret, canonical, hashlib.sha256).hexdigest()
        return {
            "X-Finh-Timestamp": str(timestamp),
            "X-Finh-Nonce": request_nonce,
            "X-Finh-Body-Sha256": digest,
            "X-Finh-Signature": signature,
        }

    def verify(self, body: bytes, headers: Mapping[str, str], *, now: int | None = None) -> None:
        try:
            timestamp = int(headers["X-Finh-Timestamp"])
            nonce = headers["X-Finh-Nonce"]
            body_digest = headers["X-Finh-Body-Sha256"]
            signature = headers["X-Finh-Signature"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SignatureError("缺少或非法的 worker 签名头") from exc
        current = int(time.time() if now is None else now)
        if abs(current - timestamp) > self._max_clock_skew_s:
            raise SignatureError("worker 请求时间戳已过期")
        digest = hashlib.sha256(body).hexdigest()
        if not hmac.compare_digest(body_digest, digest):
            raise SignatureError("worker 请求正文摘要不匹配")
        canonical = f"{timestamp}\n{nonce}\n{digest}".encode("ascii")
        expected = hmac.new(self._secret, canonical, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise SignatureError("worker 请求签名无效")
        self._discard_expired(current)
        if nonce in self._seen_nonces:
            raise ReplayError("worker 请求已被处理")
        self._seen_nonces[nonce] = timestamp + self._max_clock_skew_s

    def _discard_expired(self, now: int) -> None:
        self._seen_nonces = {
            nonce: expires_at for nonce, expires_at in self._seen_nonces.items() if expires_at >= now
        }


def allocate_artifact_dir(
    output_root: str | Path, *, user_id: str, conversation_id: str, job_id: str
) -> Path:
    """为一个任务创建唯一的 ``output/user/conversation/job`` 目录。"""
    for name, value in (("user_id", user_id), ("conversation_id", conversation_id)):
        if not _SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError(f"{name} 不是服务端允许的标识")
    if not _SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError("job_id 必须是服务端生成的 UUID 十六进制值")
    root = Path(output_root).resolve()
    target = (root / user_id / conversation_id / job_id).resolve()
    if not _is_within(target, root):  # 防御未来放宽标识格式时的路径逃逸。
        raise ValueError("任务产物目录越过了输出根")
    target.mkdir(parents=True, exist_ok=False)
    _chmod_private(target)
    return target


def extract_task_package(
    archive: str | Path,
    target_dir: str | Path,
    *,
    max_files: int = MAX_PACKAGE_FILES,
    max_uncompressed_bytes: int = MAX_PACKAGE_BYTES,
) -> list[Path]:
    """安全解包受限 ZIP；拒绝链接、路径逃逸及压缩炸弹。"""
    if max_files <= 0 or max_uncompressed_bytes <= 0:
        raise ValueError("任务包限制必须为正数")
    if Path(archive).stat().st_size > MAX_PACKAGE_BYTES:
        raise TaskPackageError("任务压缩包体积超过限制")
    destination = Path(target_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _chmod_private(destination)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [item for item in bundle.infolist() if not item.is_dir()]
            if len(members) > max_files:
                raise TaskPackageError("任务包文件数超过限制")
            if sum(item.file_size for item in members) > max_uncompressed_bytes:
                raise TaskPackageError("任务包解压后体积超过限制")
            written: list[Path] = []
            for item in members:
                if _is_link(item) or not item.filename or "\\" in item.filename:
                    raise TaskPackageError("任务包包含链接或非法路径")
                output = (destination / item.filename).resolve()
                if not _is_within(output, destination):
                    raise TaskPackageError("任务包路径越界")
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item, "r") as source, output.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=64 * 1024)
                _chmod_private(output)
                written.append(output)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise TaskPackageError(f"无法读取任务包：{exc}") from exc
    return written


def _is_link(item: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(item.external_attr >> 16)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _chmod_private(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o700)
