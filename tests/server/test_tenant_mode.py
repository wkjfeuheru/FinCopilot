"""单租户注入必须显式，且在多租户下被拒（隔离方案 P1-6）。

``create_app(data_access=...)`` 会让全体用户共用一份数据面，从而静默关闭每用户
的产物与缓存命名空间（G9）。此前那是一个不具名的副作用；现在它是一个必须写出来
的声明，并在远程监听下被拒绝。
"""

from __future__ import annotations

import pytest

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app


def make_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
        **overrides,
    )


def test_injecting_data_access_without_declaring_single_tenant_is_rejected(tmp_path):
    """不写明就注入是被拒的：那会静默关掉租户隔离。"""
    with pytest.raises(ValueError, match="single_tenant"):
        create_app(
            FakeProvider(["ok"]),
            data_access=DataAccess([]),
            settings=make_settings(tmp_path),
        )


def test_declaring_single_tenant_is_accepted(tmp_path):
    app = create_app(
        FakeProvider(["ok"]),
        data_access=DataAccess([]),
        settings=make_settings(tmp_path),
        single_tenant=True,
    )

    assert app is not None


def test_single_tenant_injection_is_rejected_when_listening_remotely(tmp_path):
    """多租户（远程监听）下共用一份数据面是隔离失效，不是配置选项。"""
    settings = make_settings(
        tmp_path,
        server={"allow_remote": True, "host": "0.0.0.0"},
        compute={"remote_worker_url": "http://worker.internal:8080"},
    )

    with pytest.raises(ValueError, match="allow_remote"):
        create_app(
            FakeProvider(["ok"]),
            data_access=DataAccess([]),
            settings=settings,
            single_tenant=True,
        )


def test_multi_tenant_mode_gives_each_user_its_own_workspace(tmp_path):
    """不开注入时，每用户仍拿到各自的命名空间。"""
    app = create_app(FakeProvider(["ok"]), settings=make_settings(tmp_path))
    scoped = app.state.scoped_settings

    assert scoped("alice").data.cache_dir != scoped("bob").data.cache_dir
    assert scoped("alice").paths.output_dir != scoped("bob").paths.output_dir
