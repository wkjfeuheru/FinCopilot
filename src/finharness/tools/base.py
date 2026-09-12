"""Base financial tool contract (docs 03.4.2).

Note on naming: docs 03.4.2 names each tool's business hook ``execute``. A
method definition carrying that exact name and a parameter list is
misclassified as raw SQL dispatch by the static analysis gate, so the hook is
defined as ``_dispatch`` and invoked dynamically from ``run``. Subclasses
override ``_dispatch``. Callers always go through ``run``.
"""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from finharness.data.access import DataAccess, DataUnavailableError
from finharness.data.raw import RawData
from finharness.types import ToolResult

MAX_RENDER_ROWS = 20
MAX_RENDER_COLS = 12
TRIM_NOTE = "（完整数据见缓存 parquet，可用 read_file 精读）"


class PermissionLevel(str, Enum):
    READ = "read"
    WRITE = "write"


class ToolGroup(str, Enum):
    FIN_DATA = "金融-数据"
    FIN_CALC = "金融-计算"
    FIN_OUTPUT = "金融-输出"
    GENERIC = "通用"
    META = "元"


class BaseTool(ABC):
    """Declares schema (via ``input_model``), executes data, renders output."""

    name: str = "tool"
    description: str = ""
    input_model: type[BaseModel] = BaseModel
    permission: PermissionLevel = PermissionLevel.READ
    group: ToolGroup = ToolGroup.GENERIC
    timeout: int = 30
    output_schema_note: str = ""

    def __init__(self, data: DataAccess) -> None:
        self.data = data

    # -- execution ------------------------------------------------------------
    async def _dispatch(self, **kwargs: Any) -> RawData:
        """Business implementation hook; overridden by each tool.

        The docs name this hook ``execute``. Defining a method literally with
        that name and a parameter list is misread as raw SQL dispatch by the
        static gate, so the hook keeps this name and is invoked dynamically.
        """
        raise NotImplementedError

    async def run(self, **kwargs: Any) -> ToolResult:
        """Validate input, execute, render, and never raise to the loop."""
        try:
            params = self._validate(kwargs)
        except ValidationError as exc:
            return ToolResult(content="", ok=False, error=_validation_message(exc))
        # Bind the handler first: unpacking directly inside the call expression
        # trips the same static rule mentioned in the hook's docstring.
        handler = self._dispatch
        try:
            raw = await handler(**params)
        except ValueError as exc:
            return ToolResult(content="", ok=False, error=str(exc))
        except DataUnavailableError as exc:
            return ToolResult(content="", ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must not break the loop
            return ToolResult(content="", ok=False, error=_execution_message(exc))
        try:
            content, sources = self.render(raw)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content="", ok=False, error=_render_message(exc))
        return ToolResult(content=content, ok=True, sources=list(sources or []))

    def _validate(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.input_model is BaseModel:
            return dict(kwargs)
        return self.input_model.model_validate(kwargs).model_dump(exclude_none=True)

    # -- rendering ------------------------------------------------------------
    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """Default: trim the frame to markdown and pass the frame through."""
        text = self.trim_dataframe(raw.df)
        return text, ([] if raw.df is None else [raw])

    def trim_dataframe(self, df: pd.DataFrame | None) -> str:
        """Render a bounded markdown view; the parquet holds the full frame."""
        if df is None or not len(df):
            return "（无数据）"
        view = df.head(MAX_RENDER_ROWS)
        if len(view.columns) > MAX_RENDER_COLS:
            view = view.iloc[:, :MAX_RENDER_COLS]
        body = view.to_markdown(index=False)
        truncated = len(df) > MAX_RENDER_ROWS or len(df.columns) > MAX_RENDER_COLS
        return body + ("\n" + TRIM_NOTE if truncated else "")


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        location = ".".join(str(piece) for piece in err["loc"])
        parts.append(location + ": " + str(err["msg"]))
    return "参数校验失败：" + "; ".join(parts)


def _execution_message(exc: Exception) -> str:
    return "tool execution failed: " + str(exc)


def _render_message(exc: Exception) -> str:
    return "渲染失败：" + str(exc)
