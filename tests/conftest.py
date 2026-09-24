"""仓库级测试公共设施（此前只有 tests/server/conftest.py）。

这里的 ``settings_with_cache`` 固化了一条几乎所有测试都要写的约定：把
``Settings.data.cache_dir`` 指向 ``tmp_path`` 下的目录，避免测试写入真实仓库的
``data_cache/``。各测试文件仍按需覆盖 context / permission / paths / ltm 等节，
但"缓存目录落在临时目录"这一条不再逐文件重抄——它曾经在 19 个文件里各写一遍，
一旦有人漏改就会污染工作区。

注意：``tests/`` 是命名空间包（无 ``__init__.py``），helper 通过
``from tests.conftest import settings_with_cache`` 复用；``pyproject.toml`` 的
``pythonpath`` 已包含仓库根，因此该导入可解析。
"""

from __future__ import annotations

from pathlib import Path

from finharness.config.settings import Settings


def settings_with_cache(tmp_path: Path, **overrides: object) -> Settings:
    """以 ``tmp_path/cache`` 为缓存目录构造 ``Settings``。

    ``overrides`` 透传给 ``Settings``；调用方显式给出 ``data=`` 时以调用方为准。
    """
    overrides.setdefault("data", {"cache_dir": tmp_path / "cache"})
    return Settings(**overrides)  # type: ignore[arg-type]
