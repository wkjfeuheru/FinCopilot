#!/usr/bin/env python
"""清理 output/ 下 ad-hoc 跑测留下的临时目录与日志，以及迁移前的旧备份。

产出物分三类，默认只清第一类：

* **output/ 内的 scratch** —— 顶层以 ``_`` 开头的目录（历史上手工覆盖
  ``cache_dir`` 跑出来的一次性缓存转储，名如 ``_t``、``_final``）、日志丢弃目录
  ``output/e2e_run/``（``pytest | tee`` / ``fin eval`` 的手工重定向），以及顶层的
  ``output/*.log``。这些都是可再生的临时数据。
* **迁移前的旧备份**（``--include-bak``）—— ``data_cache.bak/``、``state.bak/``，
  以及活跃 ``state/`` 内带日期的 ``memory.db.bak-*`` 快照。它们不被运行时使用，
  但 ``state.bak/`` 含旧 ``config.db`` / ``users.db``，删除后不可恢复。
* **评测产物**（``--include-evals``）—— ``evals/runs/``，可用 ``fin eval run`` 再生。

默认是 **dry-run**：只打印将要处理的项目与可释放字节，不删任何东西；确认真实执行
时再加 ``--apply``。``output/`` 里的真实产物（``.gitkeep``、``charts/``、按租户隔离的
``u_*/`` 目录、投研测试集 ``.md``）在内置保护名单中，任何参数组合都不会删到它们。

    python scripts/clean_output.py                       # dry-run，只看计划
    python scripts/clean_output.py --include-bak --apply # 真删（含两个 .bak 目录）
    python scripts/clean_output.py --json                # 机器可读的计划
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# output/ 下永不删除的名字（无论选中逻辑或 --keep 怎么给）。
PROTECTED_OUTPUT_NAMES = frozenset({".gitkeep", "charts", "投研问题测试集-技能路由与子代理触发.md"})
# 按租户隔离的产物目录前缀：output/<user_id>/，同样受保护。
TENANT_DIR_PREFIX = "u_"
# ad-hoc 日志丢弃目录。
LOG_DROPBOX_NAME = "e2e_run"
# --include-bak 覆盖的旧备份目录与 state/ 内带日期的 memory 快照模式。
BAK_DIR_NAMES = ("data_cache.bak", "state.bak")
STATE_SNAPSHOT_GLOB = "memory.db.bak-*"


@dataclass(frozen=True)
class Target:
    """一个待清理的目标及其展示信息。"""

    path: Path
    rel: str  # 相对仓库根的 POSIX 路径
    kind: str  # "dir" | "file"
    size: int  # 递归文件字节数
    newest: float  # 该目标内最新 mtime（epoch 秒）


def _measure(path: Path) -> tuple[int, float]:
    """返回 (递归字节数, 最新 mtime)。目录按其中所有文件累计。"""
    if path.is_file():
        stat = path.stat()
        return stat.st_size, stat.st_mtime
    total = 0
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            stat = child.stat()
        except OSError:
            continue
        if child.is_file():
            total += stat.st_size
        newest = max(newest, stat.st_mtime)
    return total, newest


def _human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _within_root(path: Path, root: Path) -> bool:
    """目标解析后必须仍落在仓库根内，避免路径逃逸。"""
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _is_protected(path: Path, root: Path) -> bool:
    """内置保护名单：只对 output/ 下的条目生效，其余交给 _within_root 把关。"""
    try:
        rel = path.resolve().relative_to(root / "output")
    except ValueError:
        return False
    if not rel.parts:
        return True  # output/ 本身
    name = rel.parts[0]
    return name in PROTECTED_OUTPUT_NAMES or name.startswith(TENANT_DIR_PREFIX)


def select_targets(
    root: Path,
    *,
    include_bak: bool,
    include_evals: bool,
    keep: frozenset[str],
    days: int,
) -> tuple[list[Target], list[str]]:
    """按规则选出待清理目标；同时返回被跳过的说明（保护名单 / 路径逃逸 / 未到期）。"""
    candidates: list[Path] = []

    output_dir = root / "output"
    if output_dir.is_dir():
        for entry in sorted(output_dir.iterdir()):
            if entry.is_dir() and (entry.name.startswith("_") or entry.name == LOG_DROPBOX_NAME):
                candidates.append(entry)
            elif entry.is_file() and entry.suffix == ".log":
                candidates.append(entry)

    if include_bak:
        for name in BAK_DIR_NAMES:
            backup = root / name
            if backup.exists():
                candidates.append(backup)
        state_dir = root / "state"
        if state_dir.is_dir():
            candidates.extend(sorted(state_dir.glob(STATE_SNAPSHOT_GLOB)))

    if include_evals:
        runs = root / "evals" / "runs"
        if runs.exists():
            candidates.append(runs)

    cutoff = None if days <= 0 else datetime.now() - timedelta(days=days)
    targets: list[Target] = []
    skipped: list[str] = []
    for path in candidates:
        if not _within_root(path, root):
            skipped.append(f"{path}：解析后越出仓库根，跳过")
            continue
        if path.is_symlink():
            skipped.append(f"{path}：符号链接，跳过")
            continue
        rel = path.relative_to(root).as_posix()
        if _is_protected(path, root) or path.name in keep or rel in keep:
            skipped.append(f"{rel}：命中保护名单，跳过")
            continue
        size, newest = _measure(path)
        if cutoff is not None and datetime.fromtimestamp(newest) >= cutoff:
            skipped.append(f"{rel}：修改时间在 {days} 天以内，跳过")
            continue
        kind = "dir" if path.is_dir() else "file"
        targets.append(Target(path=path, rel=rel, kind=kind, size=size, newest=newest))
    return targets, skipped


def _delete(target: Target) -> bool:
    try:
        if target.kind == "dir":
            shutil.rmtree(target.path)
        else:
            target.path.unlink()
    except OSError as exc:
        print(f"删除失败 {target.rel}：{exc}", file=sys.stderr)
        return False
    return True


def _plan_payload(targets: list[Target], skipped: list[str], *, applied: bool, freed: int) -> dict:
    return {
        "apply": applied,
        "targets": [
            {
                "path": t.rel,
                "kind": t.kind,
                "bytes": t.size,
                "newest": datetime.fromtimestamp(t.newest).isoformat(timespec="seconds"),
            }
            for t in targets
        ],
        "skipped": skipped,
        "total_bytes": sum(t.size for t in targets),
        "freed_bytes": freed,
    }


def _print_plan(targets: list[Target], skipped: list[str], *, apply: bool) -> None:
    total = sum(t.size for t in targets)
    mode = "删除" if apply else "将删除（dry-run）"
    print(f"\n{mode}以下 {len(targets)} 项：")
    print("=" * 72)
    print(f"{'大小':>10}  {'修改时间':<16} 路径")
    print("-" * 72)
    for t in targets:
        stamp = datetime.fromtimestamp(t.newest).strftime("%Y-%m-%d %H:%M")
        print(f"{_human_size(t.size):>10}  {stamp:<16} {t.rel}")
    print("-" * 72)
    print(f"合计 {len(targets)} 项，{_human_size(total)}")

    if skipped:
        print(f"\n跳过 {len(skipped)} 项：")
        for line in skipped:
            print(f"  - {line}")


def main(argv: list[str] | None = None) -> int:
    """打印清理计划，若加 --apply 则执行删除并汇总释放的字节数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="真实删除；缺省只做 dry-run")
    parser.add_argument("--include-bak", action="store_true", help="一并清理 data_cache.bak/、state.bak/ 与 state/memory.db.bak-*")
    parser.add_argument("--include-evals", action="store_true", help="一并清理 evals/runs/")
    parser.add_argument("--days", type=int, default=0, metavar="N", help="只清理修改时间早于 N 天的目标（0 = 不限，默认）")
    parser.add_argument("--keep", action="append", default=[], metavar="NAME", help="额外保护的名字或相对路径，可重复")
    parser.add_argument("--root", default=None, metavar="PATH", help="仓库根，默认取脚本所在目录的上一级")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出计划，便于机器消费")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else ROOT
    targets, skipped = select_targets(
        root,
        include_bak=args.include_bak,
        include_evals=args.include_evals,
        keep=frozenset(args.keep),
        days=args.days,
    )

    if not args.apply:
        if args.json:
            print(json.dumps(_plan_payload(targets, skipped, applied=False, freed=0), ensure_ascii=False, indent=2))
        else:
            _print_plan(targets, skipped, apply=False)
            print("\ndry-run：未删除任何内容。确认后加 --apply 执行。\n")
        return 0

    # --apply：JSON 模式下把人类可读输出压到 stderr，保持 stdout 是纯 JSON。
    if not args.json:
        _print_plan(targets, skipped, apply=True)

    freed = 0
    removed = 0
    for target in targets:
        if _delete(target):
            freed += target.size
            removed += 1

    if args.json:
        print(json.dumps(_plan_payload(targets, skipped, applied=True, freed=freed), ensure_ascii=False, indent=2))
    else:
        print(f"\n已删除 {removed} 项，释放 {_human_size(freed)}。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
