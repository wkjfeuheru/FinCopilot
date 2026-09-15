"""限制在产物目录与缓存目录内的文件系统工具。

``read_file`` 读取会话已经指向的文件——``output/`` 下产出的产物，或复核子代理为核验
某个数字而重读的缓存载荷。它**不是**主 Agent 用来更完整查看刚取数据的方式：那是数据
工具的 ``detail="full"``（同一次调用，更宽的渲染），因为读取 parquet 会走同一套裁剪
预算，返回的内容不会比取数时更多。``write_file`` 只有在允许的根目录内才无需确认即可
通过权限门禁。
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


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "读取会话已指向的文件（output/ 下的产物，或复核用的缓存 parquet）。"
        "若要更完整地查看刚取的数据，请用相应数据工具的 detail=\"full\"，"
        "而不是读缓存文件。"
    )
    input_model = ReadFileInput
    permission = PermissionLevel.READ
    group = ToolGroup.GENERIC
    timeout = 30

    async def _dispatch(self, *, path: str) -> RawData:
        """读取允许目录内的文件；parquet 读为数据框，其余按文本读取（截断到 MAX_READ_BYTES）。"""
        settings = self.data.settings
        roots = [Path(settings.paths.output_dir).resolve(), (Path(settings.data.cache_dir) / "parquet").resolve()]
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
        """将内容写入允许目录内的文件（自动创建父目录），返回写入路径与字符数。"""
        settings = self.data.settings
        roots = [Path(settings.paths.output_dir).resolve(), (Path(settings.data.cache_dir) / "parquet").resolve()]
        target = _resolve_within(path, roots)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return RawData(kind="text", text=f"已写入 {target}（{len(content)} 字符）",
                       endpoint="file:write", params={"path": str(target), "bytes": len(content)})
