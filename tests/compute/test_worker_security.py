import io
import zipfile

import pytest

from finharness.compute.protocol import (
    ReplayError,
    SignatureError,
    TaskPackageError,
    TaskSigner,
    allocate_artifact_dir,
    extract_task_package,
)


def test_signed_task_rejects_tampering_and_replay():
    signer = TaskSigner("test-secret-at-least-16-bytes", max_clock_skew_s=60)
    headers = signer.sign(b'{"kind":"chart"}', now=100)

    signer.verify(b'{"kind":"chart"}', headers, now=101)
    with pytest.raises(ReplayError):
        signer.verify(b'{"kind":"chart"}', headers, now=101)

    fresh = signer.sign(b'{"kind":"chart"}', now=100, nonce="other")
    with pytest.raises(SignatureError):
        signer.verify(b'{"kind":"backtest"}', fresh, now=101)


def test_task_zip_rejects_path_traversal_and_symlinks(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")

    with pytest.raises(TaskPackageError):
        extract_task_package(archive, tmp_path / "target")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(symlink, "w") as bundle:
        bundle.writestr(info, "elsewhere")

    with pytest.raises(TaskPackageError):
        extract_task_package(symlink, tmp_path / "target-2")


def test_artifacts_are_scoped_to_server_generated_ids(tmp_path):
    path = allocate_artifact_dir(
        tmp_path / "output", user_id="u_123", conversation_id="c_456", job_id="a" * 32
    )
    assert path == tmp_path / "output" / "u_123" / "c_456" / ("a" * 32)
    assert path.exists()

    with pytest.raises(ValueError):
        allocate_artifact_dir(
            tmp_path / "output", user_id="../other", conversation_id="c_456", job_id="a" * 32
        )
