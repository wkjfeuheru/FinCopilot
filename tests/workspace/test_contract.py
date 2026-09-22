"""workspace 契约：一个租户能触及什么、不能触及什么（隔离方案 P1-1/P1-3）。

这组测试就是 P1-1 的"契约"本身——它把可达性写成可执行的断言，而不是文档里的
一句话。价值在于：可达性收敛到单一来源之后，**由本文件守住它**，任何一处工具或
端点想私自放宽范围，都会在这里失败。
"""

from __future__ import annotations

import pytest

from finharness.config.settings import Settings
from finharness.workspace import Workspace, WorkspaceViolation


def make_workspace(tmp_path) -> Workspace:
    return Workspace(
        Settings(
            data={"cache_dir": tmp_path / "data_cache"},
            paths={
                "output_dir": tmp_path / "output",
                "state_dir": tmp_path / "state",
                "memory_db": tmp_path / "state" / "memory.db",
                "auth_db": tmp_path / "state" / "users.db",
                "config_db": tmp_path / "state" / "config.db",
                "secret_key": tmp_path / "state" / "secret.key",
            },
        )
    )


# -- 可读范围 ------------------------------------------------------------------

def test_artifacts_under_output_are_readable(tmp_path):
    workspace = make_workspace(tmp_path)
    assert workspace.is_readable(workspace.output_dir / "report.md")


def test_cached_payloads_are_readable(tmp_path):
    """复核子代理要按句柄重读载荷，因此缓存的两棵子树必须可读。"""
    workspace = make_workspace(tmp_path)

    assert workspace.is_readable(workspace.cache_parquet / "2026-09" / "abc.parquet")
    assert workspace.is_readable(workspace.cache_pdf / "doc.pdf")


def test_the_cache_index_is_not_readable(tmp_path):
    """cache 根下的 index.db 是跨租户的查找索引，不属于"按句柄取回的材料"。"""
    workspace = make_workspace(tmp_path)

    assert not workspace.is_readable(workspace.cache_dir / "index.db")


def test_resolve_read_accepts_a_path_inside_the_workspace(tmp_path):
    workspace = make_workspace(tmp_path)
    target = workspace.output_dir / "a.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x", encoding="utf-8")

    assert workspace.resolve_read(str(target)) == target.resolve()


# -- 不可读范围（契约的要害）---------------------------------------------------

def test_state_files_are_not_readable(tmp_path):
    """密钥与租户数据库必须不可达——这是 P0-3 搬到 state/ 的全部意义。"""
    workspace = make_workspace(tmp_path)

    for name in ("secret.key", "users.db", "memory.db", "config.db"):
        assert not workspace.is_readable(workspace.state_dir / name), name


def test_resolve_read_rejects_a_state_file(tmp_path):
    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceViolation):
        workspace.resolve_read(str(workspace.state_dir / "users.db"))


def test_resolve_read_rejects_a_traversal_escape(tmp_path):
    workspace = make_workspace(tmp_path)
    escaped = workspace.output_dir / ".." / ".." / "etc" / "passwd"

    with pytest.raises(WorkspaceViolation):
        workspace.resolve_read(str(escaped))


def test_resolve_read_rejects_a_symlink_pointing_outside(tmp_path):
    """持久化符号链接必须被拒：resolve() 展开后落在树外。"""
    workspace = make_workspace(tmp_path)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    link = workspace.output_dir / "sneaky.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("该平台不允许创建符号链接（Windows 需管理员）")

    with pytest.raises(WorkspaceViolation):
        workspace.resolve_read(str(link))


def test_resolve_read_rejects_junk_input(tmp_path):
    workspace = make_workspace(tmp_path)

    for bad in ("", "   ", "\x00bad"):
        with pytest.raises(WorkspaceViolation):
            workspace.resolve_read(bad)


# -- 可写范围 ------------------------------------------------------------------

def test_only_output_is_writable(tmp_path):
    workspace = make_workspace(tmp_path)

    assert workspace.is_writable(workspace.output_dir / "report.md")
    assert not workspace.is_writable(workspace.cache_parquet / "poison.parquet")
    assert not workspace.is_writable(workspace.cache_pdf / "x.pdf")
    assert not workspace.is_writable(workspace.state_dir / "users.db")


def test_resolve_write_rejects_the_cache(tmp_path):
    """缓存拒写：lookup 键共享，模型侧一次写入即投毒他人将命中的载荷。"""
    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceViolation):
        workspace.resolve_write(str(workspace.cache_dir / "parquet" / "p.parquet"))


def test_writable_roots_are_a_subset_of_readable_roots(tmp_path):
    """写面不得宽于读面：任何可写位置都必须先可读。"""
    workspace = make_workspace(tmp_path)

    for root in workspace.writable_roots():
        assert workspace.is_readable(root)


# -- 门禁用的判定 --------------------------------------------------------------

def test_is_artifact_write_matches_resolve_write(tmp_path):
    """门与工具必须同源：二者对"这是不是产物写入"的结论不得分叉。"""
    workspace = make_workspace(tmp_path)
    inside = str(workspace.output_dir / "ok.md")
    cache = str(workspace.cache_parquet / "p.parquet")

    assert workspace.is_artifact_write(inside) is True
    assert workspace.resolve_write(inside) == (workspace.output_dir / "ok.md").resolve()
    assert workspace.is_artifact_write(cache) is False


def test_is_in_cache_covers_the_whole_cache_tree(tmp_path):
    """拒写面比可读面宽：缓存整棵树都不许写，包括只读开放之外的部分。"""
    workspace = make_workspace(tmp_path)

    assert workspace.is_in_cache(str(workspace.cache_dir / "index.db"))
    assert workspace.is_in_cache(str(workspace.cache_parquet / "p.parquet"))
    assert workspace.is_in_cache(str(workspace.output_dir / "ok.md")) is False


def test_judgements_are_false_for_junk_input(tmp_path):
    """非法输入与越界输入走同一条拒绝路径，不抛异常。"""
    workspace = make_workspace(tmp_path)

    assert workspace.is_in_cache(None) is False
    assert workspace.is_artifact_write("") is False


# -- P1-2 的输入：挂载清单 ------------------------------------------------------

def test_mount_plan_marks_state_and_cache_root_as_unmounted(tmp_path):
    """P1-2 按此清单挂载：清单之外一律不挂载，于是密钥不是被"检查"挡住，
    而是根本不在命名空间里。
    """
    workspace = make_workspace(tmp_path)
    plan = dict(workspace.mount_plan())

    assert plan[workspace.output_dir] == "rw"
    assert plan[workspace.cache_parquet] == "ro"
    assert plan[workspace.cache_pdf] == "ro"
    # 密钥目录与整个缓存根都不在挂载清单里。
    assert workspace.state_dir not in plan
    assert workspace.cache_dir not in plan


def test_mount_plan_is_derived_from_the_root_lists(tmp_path):
    """挂载清单由读写根推导，不另行书写——否则契约改了它会漂移。"""
    workspace = make_workspace(tmp_path)
    plan = dict(workspace.mount_plan())

    assert set(plan) == set(workspace.readable_roots()) | set(workspace.writable_roots())
