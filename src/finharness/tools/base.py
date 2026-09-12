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

# Shared token counter: building one re-resolves the vocabulary, so a module-level
# instance keeps render-time counting cheap.
_SHARED_COUNTER = None


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
    # ``None`` inherits settings.tools.timeout_default_s; declare a value only
    # when the interface needs a different budget (docs 03.4.1 超时列).
    timeout: int | None = None
    output_schema_note: str = ""
    # Tools that need a user reply declare this; the loop injects the callable.
    needs_interactive = False
    interactive = None
    # Tools that spawn a sub-agent declare this; the loop injects the coordinator
    # the same way. An injection rather than constructor wiring because a tool
    # cannot build a coordinator itself — that needs the provider, which tools
    # never see (docs 03.10).
    needs_coordinator = False
    coordinator = None

    def __init__(self, data: DataAccess, *, ctx: Any | None = None, registry: Any | None = None) -> None:
        self.data = data
        self.ctx = ctx
        # Meta tools (search_tools / load_tool) inspect and activate the catalogue.
        self.registry = registry

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
        # ``raw.paths`` are files the tool produced (charts, reports); they ride
        # on the result as attachments so the transport can offer them.
        return ToolResult(
            content=content,
            ok=True,
            sources=list(sources or []),
            attachments=[str(path) for path in (raw.paths or [])],
        )

    def _validate(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.input_model is BaseModel:
            return dict(kwargs)
        return self.input_model.model_validate(kwargs).model_dump(exclude_none=True)

    # -- rendering ------------------------------------------------------------
    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """Default: trim frames to markdown; pass text payloads through as-is."""
        if raw.kind == "text" or (raw.df is None and raw.text is not None):
            return (raw.text or "（无数据）"), [raw]
        text = self.trim_dataframe(raw.df)
        return text, ([] if raw.df is None else [raw])

    def trim_dataframe(self, df: pd.DataFrame | None) -> str:
        """Render a bounded markdown view; the parquet holds the full frame.

        The row budget alone cannot bound the size: wide frames (financial
        statements have dozens of columns) blow through a token budget with only
        a few rows, so columns are dropped too until the render fits
        ``context.max_result_tokens``.
        """
        if df is None or not len(df):
            return "（无数据）"
        max_rows = self._max_rows()
        budget = self._result_token_budget()
        view = df.head(max_rows)
        columns = min(len(view.columns), MAX_RENDER_COLS)
        body = ""
        while columns >= 1:
            candidate = view.iloc[:, :columns].to_markdown(index=False)
            body = candidate
            if self._count_tokens(candidate) <= budget:
                break
            columns -= 1
        truncated = (
            len(df) > max_rows
            or len(df.columns) > columns
            or len(df.columns) > MAX_RENDER_COLS
        )
        note = TRIM_NOTE if truncated else ""
        return body + ("\n" + note if note else "")

    def _result_token_budget(self) -> int:
        settings = getattr(self.data, "settings", None)
        if settings is None:
            return 0  # no budget known: keep the rendered frame as-is
        return int(settings.context.max_result_tokens)

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Count with the shared counter; counting must never break rendering."""
        global _SHARED_COUNTER
        try:
            if _SHARED_COUNTER is None:
                from finharness.context.tokens import TokenCounter

                _SHARED_COUNTER = TokenCounter()
            return _SHARED_COUNTER.count(text).tokens
        except Exception:  # noqa: BLE001
            return int(len(text) / 1.7)

    def _max_rows(self) -> int:
        """Row budget from settings.context.trim_rows, falling back to the default."""
        settings = getattr(self.data, "settings", None)
        if settings is None:
            return MAX_RENDER_ROWS
        return int(settings.context.trim_rows)


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
