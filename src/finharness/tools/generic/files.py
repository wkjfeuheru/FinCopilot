"""Filesystem tools restricted to the artefact and cache directories.

``read_file`` is what the data tools point at when they say "full data lives in
the parquet". ``write_file`` only passes the gate without confirmation inside
``output/``; anywhere else it needs approval.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup

MAX_READ_BYTES = 200_000


class ReadFileInput(BaseModel):
    path: str = Field(description="待读取的文件路径（限 output/ 与 data_cache/ 目录内）")


class WriteFileInput(BaseModel):
    path: str = Field(description="写入路径；output/ 内免确认，其余目录需用户确认")
    content: str = Field(description="写入内容")


def _resolve_within(raw: str, roots: list[Path]) -> Path:
    """Resolve a path and require it to stay inside one of the allowed roots."""
    target = Path(raw).resolve()
    for root in roots:
        try:
            if target == root or target.is_relative_to(root):
                return target
        except (OSError, ValueError):
            continue
    allowed = "、".join(str(root) for root in roots)
    raise ValueError(f"路径超出允许范围（仅限 {allowed}）：{raw}")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取 output/ 或 data_cache/ 目录内的文件（如缓存 parquet 的完整数据）。"
    input_model = ReadFileInput
    permission = PermissionLevel.READ
    group = ToolGroup.GENERIC
    timeout = 30

    async def _dispatch(self, *, path: str) -> RawData:
        settings = self.data.settings
        roots = [Path(settings.paths.output_dir).resolve(), Path(settings.data.cache_dir).resolve()]
        target = _resolve_within(path, roots)
        if not target.is_file():
            raise ValueError(f"文件不存在：{path}")
        suffix = target.suffix.lower()
        if suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(target)
            return RawData(kind="df", df=df, endpoint=f"file:{target.name}",
                           params={"path": str(target)}, parquet_path=str(target))
        text = target.read_text(encoding="utf-8", errors="replace")[:MAX_READ_BYTES]
        return RawData(kind="text", text=text, endpoint=f"file:{target.name}",
                       params={"path": str(target)})


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "写入文件；output/ 目录内直接写入，其他路径需用户确认。"
    input_model = WriteFileInput
    permission = PermissionLevel.WRITE
    group = ToolGroup.GENERIC
    timeout = 30

    async def _dispatch(self, *, path: str, content: str) -> RawData:
        settings = self.data.settings
        roots = [Path(settings.paths.output_dir).resolve(), Path(settings.data.cache_dir).resolve()]
        target = _resolve_within(path, roots)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return RawData(kind="text", text=f"已写入 {target}（{len(content)} 字符）",
                       endpoint="file:write", params={"path": str(target), "bytes": len(content)})
