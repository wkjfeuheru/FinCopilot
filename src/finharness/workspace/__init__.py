"""workspace：一个租户的**可达文件视图**（隔离方案 P1-1 / P1-3）。

**为什么需要这一层。** 在此之前，"哪些路径可达"由六处各自拼装：``read_file``/
``write_file`` 的读取与写入根、``read_pdf``、``summarize_document``、产物下载
端点，以及权限门的白名单与拒写判定。它们大体一致，但是**各自书写**的——于是

* 任何一处写错都是一个绕过点，而修一处不等于修全部；
* 各工具的"可达范围"会无意识地漂移（例如 ``read_pdf`` 能读 ``cache/pdf`` 而
  ``read_file`` 不能），漂移本身既可能是 bug，也可能让某个工具成为另一个工具
  的绕过通道。

这里把可达性收敛成**唯一来源**：工具、端点与权限门都只能问它，不再自己拼路径。

**这也是 Phase 1 的契约所在。** 当前的强制手段仍是应用层检查（``resolve()``
后判包含）。真正的边界应由命名空间提供：把 :attr:`Workspace.output_dir` 与
:attr:`Workspace.cache_dir` 以读写/只读挂载进一个受限执行域，其余一律不挂载，
则"路径不可达"就不再依赖任何一段 Python 检查是否正确。把可达性收敛到这一处，
正是让那次迁移只改一个地方、且能被本模块的契约测试守住的前提。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["Workspace", "WorkspaceViolation"]


class WorkspaceViolation(ValueError):
    """请求的路径超出该租户的可达范围。

    继承 ``ValueError``：工具层一直以 ``ValueError`` 表达"路径不允许"，循环据此
    生成结构化失败结果。保持这个基类，使收敛改造不改变任何既有的错误路径行为。
    """


class Workspace:
    """一个租户在读/写上分别能触及什么。

    契约（有意写得可断言，测试直接守住它）：

    * **可读**：``output_dir/``、``cache_dir/parquet/``、``cache_dir/pdf/``。
    * **可写**：仅 ``output_dir/``。
    * **不可达**：``state_dir/``（主密钥、用户库、对话库、加密配置库）、
      ``cache_dir/`` 根本身（含 ``index.db``——那是全部租户的 lookup 索引）。
    * 缓存子树只读：缓存键是共享的，模型侧一次写入即可替换他人稍后命中的载荷
      （缓存投毒），而模型没有正当理由手改 parquet。
    """

    def __init__(self, settings) -> None:
        self.output_dir = Path(settings.paths.output_dir).resolve()
        self.cache_dir = Path(settings.data.cache_dir).resolve()
        self.state_dir = Path(settings.paths.state_dir).resolve()
        # 缓存里对 agent 开放的两棵子树。刻意只到子目录一级：cache 根下的
        # index.db 是跨租户的查找索引，不属于任何"按句柄取回的材料"。
        self.cache_parquet = (self.cache_dir / "parquet").resolve()
        self.cache_pdf = (self.cache_dir / "pdf").resolve()

    # -- 契约 ----------------------------------------------------------------
    def readable_roots(self) -> tuple[Path, ...]:
        """agent 可读的全部根（按固定顺序，便于稳定报错）。"""
        return (self.output_dir, self.cache_parquet, self.cache_pdf)

    def writable_roots(self) -> tuple[Path, ...]:
        """agent 可写的全部根。缓存刻意不在其中。"""
        return (self.output_dir,)

    def forbidden_roots(self) -> tuple[Path, ...]:
        """必须不可达的位置；由契约测试守门，并作为 P1-2 挂载清单的排除项。"""
        return (self.state_dir, self.cache_dir)

    # -- 判定 ----------------------------------------------------------------
    def is_readable(self, target: Path) -> bool:
        return any(_within(target, root) for root in self.readable_roots())

    def is_writable(self, target: Path) -> bool:
        return any(_within(target, root) for root in self.writable_roots())

    def is_artifact_write(self, raw: str) -> bool:
        """``raw`` 是否指向可写区（供权限门判断"产物写入、免确认"）。

        与 :meth:`resolve_write` 同源：门与工具因此不可能对"这是不是产物写入"
        得出不同结论——此前它们各自拼根，正是这种不一致的温床。
        """
        target = _try_resolve(raw)
        return target is not None and self.is_writable(target)

    def is_in_cache(self, raw: str) -> bool:
        """``raw`` 是否落在缓存区（供权限门结构性拒写）。

        注意用的是整个 ``cache_dir`` 而非两个可读子目录：写入的拒绝面应当宽于
        读取的开放面，宁可多拒一个也不让缓存被改写。
        """
        target = _try_resolve(raw)
        return target is not None and _within(target, self.cache_dir)

    # -- 解析 ----------------------------------------------------------------
    def resolve_read(self, raw: str) -> Path:
        """解析一个待读路径，越界即 :class:`WorkspaceViolation`。"""
        return self._resolve(raw, self.readable_roots(), "读取")

    def resolve_write(self, raw: str) -> Path:
        """解析一个待写路径，越界即 :class:`WorkspaceViolation`。"""
        return self._resolve(raw, self.writable_roots(), "写入")

    def _resolve(self, raw: str, roots: tuple[Path, ...], verb: str) -> Path:
        target = _try_resolve(raw)
        if target is not None and any(_within(target, root) for root in roots):
            return target
        allowed = "、".join(str(root) for root in roots)
        raise WorkspaceViolation(
            f"路径超出允许范围（{verb}仅限 {allowed}）：{raw}"
        )

    # -- 供跨进程边界使用的挂载清单（P1-2）------------------------------------
    def mount_plan(self) -> tuple[tuple[Path, str], ...]:
        """受限执行域应当挂载什么、以什么权限。

        返回 ``(路径, "rw"|"ro")``。这是 P1-1 契约对 P1-2 的直接产出：执行沙箱
        按此清单挂载，**清单之外一律不挂载**，于是 ``forbidden_roots()`` 里的
        东西不是被检查挡住，而是根本不在命名空间里。

        由 :meth:`readable_roots`/:meth:`writable_roots` 推导而非另行书写，
        因此契约改了、挂载清单跟着改，不会各自漂移。
        """
        plan: dict[Path, str] = {}
        for root in self.readable_roots():
            plan[root] = "ro"
        for root in self.writable_roots():
            plan[root] = "rw"
        return tuple(plan.items())


def _try_resolve(raw: str) -> Path | None:
    """解析路径；非法输入（空、含 NUL、超长等）返回 None 而不是抛出。

    返回 None 让调用方统一走"不在允许范围"的拒绝路径，避免非法输入与越界
    输入产生两种不同的对外行为。
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return Path(raw).resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def _within(target: Path, root: Path) -> bool:
    """``target`` 是否等于 ``root`` 或位于其下（两者均已解析）。

    ``resolve()`` 已展开符号链接，因此指向树外的软链在这里会被判为越界；
    ``..`` 逃逸同样在解析阶段被消解。
    """
    try:
        return target == root or target.is_relative_to(root)
    except (OSError, ValueError):
        return False
