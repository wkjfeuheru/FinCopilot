"""租户级数据生命周期：导出（可携带权）与抹除（遗忘权）。

隔离方案 Phase 2 要求"租户级导出/删除必须覆盖缓存与审计"。记忆数据库只
是其中一处——同一份租户数据还散落在三个地方，漏掉任何一处都会得到一份
"看起来删干净了"的假象：

* ``output/<user_id>/``：该用户的产物（报告、图表）；
* ``cache/users/<user_id>/``：该用户的缓存载荷（按租户命名空间，见 P0-1）；
* 审计 JSONL：共享一个进程级文件，按 ``user_id`` 字段分流。

抹除按"先记后删"的顺序进行：先把语义条目取出（向量库要按 id 清理点），再删
数据库行，再删文件，最后改写审计文件。任何一步失败都向上抛，不做静默降级——
遗忘权"部分完成"不是可接受的结果。
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from finharness.context.memory.store import MemoryStore


def _user_output_dir(settings: Any, user_id: str) -> Path:
    return Path(settings.paths.output_dir) / user_id


def _user_cache_dir(settings: Any, user_id: str) -> Path:
    return Path(settings.data.cache_dir) / "users" / user_id


def export_tenant_archive(
    *,
    store: MemoryStore,
    user_id: str,
    settings: Any,
) -> bytes:
    """打包该用户的全部数据为一个 ZIP：``userdata.json`` + 产物 + 缓存。

    产物与缓存按原目录结构收进 ``output/`` 与 ``cache/`` 前缀下，使导出既能
    用程序解析（``userdata.json``），也能直接翻阅文件。空目录安全。
    """
    memory = store.export_user_data(user_id=user_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "userdata.json",
            json.dumps(memory, ensure_ascii=False, indent=2, default=str),
        )
        for prefix, root in (
            ("output", _user_output_dir(settings, user_id)),
            ("cache", _user_cache_dir(settings, user_id)),
        ):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    bundle.write(path, f"{prefix}/{path.relative_to(root).as_posix()}")
    return buffer.getvalue()


def purge_tenant_data(
    *,
    store: MemoryStore,
    user_id: str,
    settings: Any,
    audit_path: str | Path,
    semantic_index: Any | None = None,
) -> dict[str, Any]:
    """抹除该租户的一切数据，覆盖记忆库、向量库、产物、缓存与审计。

    返回一个可审计的摘要（各表删除行数与文件/审计行计数），使运维能核对
    这次抹除实际动了什么，而不是只能相信它"没有报错"。
    """

    def facts_for_vectors() -> list[Any]:
        if semantic_index is None:
            return []
        # 先把条目取出：删除后就没有 fa_uid 可用来清理向量库的点了。
        return store.list_ltm_facts(user_id=user_id, limit=1_000_000)

    facts = facts_for_vectors()
    deleted = store.purge_user_data(user_id=user_id)

    if semantic_index is not None:
        for fact in facts:
            semantic_index.unindex_fact(fact, user_id=user_id)

    removed_files = 0
    for root in (_user_output_dir(settings, user_id), _user_cache_dir(settings, user_id)):
        if root.exists():
            removed_files += sum(1 for path in root.rglob("*") if path.is_file())
            shutil.rmtree(root, ignore_errors=False)

    audit_removed = _strip_audit_lines(Path(audit_path), user_id=user_id)

    return {
        "user_id": user_id,
        "deleted_rows": deleted,
        "unindexed_facts": len(facts),
        "removed_files": removed_files,
        "removed_audit_lines": audit_removed,
    }


def _strip_audit_lines(path: Path, *, user_id: str) -> int:
    """从审计 JSONL 中移除该用户的行；原子替换，失败不损坏原文件。

    审计是共享文件，因此这里**改写**而非删除整个文件。无法解析的行原样保留
    ——宁可能留下一条读不动的记录，也不因一条坏行丢掉他人的审计。
    """
    if not path.is_file():
        return 0
    kept: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if isinstance(record, dict) and record.get("user_id") == user_id:
            removed += 1
        else:
            kept.append(line)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    temporary.replace(path)
    return removed
