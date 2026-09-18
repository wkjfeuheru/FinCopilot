"""限制在产物目录与缓存目录内的文件系统工具。

``read_file`` 读取会话已经指向的文件——``output/`` 下产出的产物，或复核子代理为核验
某个数字而重读的缓存载荷。它**不是**主 Agent 用来更完整查看刚取数据的方式：那是数据
工具的 ``detail="full"``（同一次调用，更宽的渲染），因为读取 parquet 会走同一套裁剪
预算，返回的内容不会比取数时更多。``write_file`` 只有在允许的根目录内才无需确认即可
通过权限门禁。

**所有文件 I/O 都在线程中执行。** 循环用 ``asyncio.wait_for`` 施加工具超时，而它只能
取消会在 await 点让出的协程：直接在 async 函数里做同步 ``read_text``/``read_parquet``
会让超时形同虚设，并在阻塞期间占住整个事件循环——多租户下一个大文件即可让所有租户
一起停顿。此外读写各有显式的体积上限，使"多大算过界"由配置决定而非由输入决定。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool

MAX_READ_BYTES = 200_000
# 写入上限。与读取的 200_000 字符对齐：允许写的比允许读回的多没有意义，
# 而超过这个体量的内容本就该由 write_report 之类的专门工具产出。
MAX_WRITE_BYTES = 200_000
# parquet 需要单独的字节上限：它按数据框读取，不经过文本截断那条路径，
# 一个超大文件会在解码阶段就吃掉大量内存。
MAX_PARQUET_BYTES = 64 * 1024 * 1024


def _resolve_within(raw: str, roots: list[Path]) -> Path:
    """解析路径并要求其保持在某个允许的根目录之内。"""
    target = Path(raw).resolve()
    for root in roots:
        try:
            if target == root or target.is_relative_to(root):
                return target
        except (OSError, ValueError):
            continue
    allowed = "、".join(str(root) for root in roots)
    raise ValueError(f"路径超出允许范围（仅限 {allowed}）：{raw}")


def _roots_for(settings) -> list[Path]:
    """读取类工具的允许根：产物目录与该用户的缓存 parquet 子树。"""
    return [
        Path(settings.paths.output_dir).resolve(),
        (Path(settings.data.cache_dir) / "parquet").resolve(),
    ]


def _write_root_for(settings) -> Path:
    """写类工具的唯一允许根：产物目录。

    缓存 parquet 子树对读取开放（复核子代理按句柄重读载荷），但对写入
    关闭：缓存的 lookup 键对所有用户共享，一次模型侧的写入就能替换另一
    个用户稍后命中的载荷——这是缓存投毒，而不是产物产出。
    """
    return Path(settings.paths.output_dir).resolve()


@tool(
    name="read_file",
    description=(
        "读取会话已指向的文件（output/ 下的产物，或复核用的缓存 parquet）。"
        "若要更完整地查看刚取的数据，请用相应数据工具的 detail=\"full\"，"
        "而不是读缓存文件。"
    ),
    capability=Capability.FILE,
    group=ToolGroup.GENERIC,
    timeout=30,
)
class ReadFileTool(BaseTool):
    @param("path", desc="待读取的文件路径（限 output/ 与 data_cache/ 目录内）")
    async def _dispatch(self, *, path: str) -> RawData:
        """读取允许目录内的文件；parquet 读为数据框，其余按文本读取（截断到 MAX_READ_BYTES）。"""
        settings = self.data.settings
        target = _resolve_within(path, _roots_for(settings))
        if not await asyncio.to_thread(target.is_file):
            raise ValueError(f"文件不存在：{path}")
        suffix = target.suffix.lower()
        if suffix == ".parquet":
            return await asyncio.to_thread(self._read_parquet, target)
        text = await asyncio.to_thread(self._read_text, target)
        return RawData(kind="text", text=text, endpoint=f"file:{target.name}",
                       params={"path": str(target)})

    @staticmethod
    def _read_text(target: Path) -> str:
        """读取文本并截断。先看大小再读，避免把超大文件整个读进内存。"""
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise ValueError(f"无法读取文件：{target.name}") from exc
        if size > MAX_READ_BYTES * 4:
            # 远大于截断上限时只读前若干字节即可，无需载入全文。
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                return handle.read(MAX_READ_BYTES)
        return target.read_text(encoding="utf-8", errors="replace")[:MAX_READ_BYTES]

    @staticmethod
    def _read_parquet(target: Path) -> RawData:
        import pandas as pd

        try:
            size = target.stat().st_size
        except OSError as exc:
            raise ValueError(f"无法读取文件：{target.name}") from exc
        if size > MAX_PARQUET_BYTES:
            raise ValueError(
                f"parquet 文件过大（{size} 字节，上限 {MAX_PARQUET_BYTES}）；"
                "请改用相应数据工具的 detail=\"full\""
            )
        df = pd.read_parquet(target)
        return RawData(kind="df", df=df, endpoint=f"file:{target.name}",
                       params={"path": str(target)}, parquet_path=str(target))


@tool(
    name="write_file",
    description="写入文件；仅限 output/ 目录内（缓存目录不可写）。",
    capability=Capability.FILE,
    group=ToolGroup.GENERIC,
    permission="write",
    timeout=30,
)
class WriteFileTool(BaseTool):
    @param("path", desc="写入路径；仅限 output/ 目录内")
    @param("content", desc="写入内容")
    async def _dispatch(self, *, path: str, content: str) -> RawData:
        """将内容写入允许目录内的文件（自动创建父目录），返回写入路径与字符数。"""
        settings = self.data.settings
        target = _resolve_within(path, [_write_root_for(settings)])
        if len(content) > MAX_WRITE_BYTES:
            raise ValueError(
                f"写入内容过大（{len(content)} 字符，上限 {MAX_WRITE_BYTES}）"
            )
        await asyncio.to_thread(self._write, target, content)
        return RawData(kind="text", text=f"已写入 {target}（{len(content)} 字符）",
                       endpoint="file:write", params={"path": str(target), "bytes": len(content)})

    @staticmethod
    def _write(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
