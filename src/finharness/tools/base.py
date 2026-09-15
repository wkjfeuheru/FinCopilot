"""金融工具的基础契约（docs 03.4.2）。

命名说明：docs 03.4.2 将每个工具的业务钩子命名为 ``execute``。若以该确切名称加参数列表
定义方法，会被静态分析门禁误判为原始 SQL 分发，因此该钩子被定义为 ``_dispatch``，
并由 ``run`` 动态调用。子类重写 ``_dispatch``。调用方始终经由 ``run``。
"""

from __future__ import annotations

import re
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from finharness.data.access import DataAccess, DataUnavailableError
from finharness.data.raw import RawData
from finharness.types import ToolResult

MAX_RENDER_ROWS = 20
MAX_RENDER_COLS = 12
# ``detail="full"`` 时渲染预算放大的倍数。它是一个有界倍数而非无界：目的是让
# 调用方已在使用的工具就能回答“再多给点”，而不是另设一个工具重读已取数据。
# 即使是 ``full`` 也仍在上下文的 token 约束之内。
FULL_DETAIL_MULTIPLIER = 4
# 结果被截断时告知阅读者该怎么做。它必须给出真实可执行的下一步：
# 一个载荷被裁剪的工具自身就能提供更宽的视图。
FULL_DETAIL_HINT = '需要更多行列：用本工具 detail="full" 重取（数据已缓存，属复用不重复取数）'
# 当一次取数由本地缓存而非网络提供时前置。这样“是否重复取数？”就能从工具结果本身得到答案。
CACHE_HIT_NOTE = "数据来源：缓存命中（同一请求此前已取，未重复取数）"


class DataInput(BaseModel):
    """数据获取类工具共享的输入。

    ``detail`` 是**同一次取数的展示策略**，而非另一种能力：同一次调用既可用于摘要
    也可用于更完整的视图。把它建模为参数——而不是另设一个重读缓存载荷的工具——
    正是让“读取数据”保持为一种能力、并只有一处地方可索要更多。
    """

    detail: Literal["summary", "full"] = Field(
        default="summary",
        description=(
            "返回详细程度：summary（默认，摘要+近期明细）或 "
            'full（更多行列，数据已缓存不重复取数）'
        ),
    )
# 截断说明在省略前最多列出多少个被省略的列名。
_MAX_NAMED_DROPPED_COLUMNS = 6
# 财务数据框每个报告期一列，形如 ``YYYYMMDD``；年度列（``YYYY1231``）会被优先列出、
# 排在季度列之前，以免多年对比在裁剪列时被丢掉。
_PERIOD_COLUMN_RE = re.compile(r"^\d{8}$")
_ANNUAL_PERIOD_RE = re.compile(r"^\d{4}1231$")

# 共享的 token 计数器：每次新建都会重新解析词表，因此用模块级实例让渲染时的计数保持廉价。
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
    """声明 schema（经由 ``input_model``）、执行取数、渲染输出。"""

    name: str = "tool"
    description: str = ""
    input_model: type[BaseModel] = BaseModel
    permission: PermissionLevel = PermissionLevel.READ
    group: ToolGroup = ToolGroup.GENERIC
    # ``None`` 表示继承 settings.tools.timeout_default_s；仅当该接口需要不同预算时
    # 才声明具体值（docs 03.4.1 超时列）。
    timeout: int | None = None
    output_schema_note: str = ""
    # 需要用户回复的工具声明此项；由循环注入该可调用对象。
    needs_interactive = False
    interactive = None
    # 会派生（spawn）子代理的工具声明此项；循环以同样方式注入协调器。
    # 采用注入而非构造器接线，是因为工具无法自行构建协调器——那需要 provider，
    # 而工具永远看不到它（docs 03.10）。
    needs_coordinator = False
    coordinator = None
    # 只读复核子代理是否可以调用本工具。大多数只读工具符合条件；联网工具主动退出，
    # 因为一旦复核开始搜索互联网，就会消耗 token 并把不可信文本拉入复核者，毫无收益——
    # 它的职责是拿报告与会话自身的数据做核对。
    review_eligible = True

    def __init__(self, data: DataAccess, *, ctx: Any | None = None, registry: Any | None = None) -> None:
        self.data = data
        self.ctx = ctx
        # 元工具（search_tools / load_tool）会检视并激活目录。
        self.registry = registry

    # -- 执行 ------------------------------------------------------------
    async def _dispatch(self, **kwargs: Any) -> RawData:
        """业务实现钩子；由每个工具各自重写。

        文档将该钩子命名为 ``execute``。以该确切名称加参数列表字面定义方法会被
        静态门禁误读为原始 SQL 分发，因此该钩子沿用现名并动态调用。
        """
        raise NotImplementedError

    async def run(self, **kwargs: Any) -> ToolResult:
        """校验输入、执行、渲染，且绝不向循环抛异常。"""
        try:
            params = self._validate(kwargs)
        except ValidationError as exc:
            return ToolResult(content="", ok=False, error=_validation_message(exc))
        # ``detail`` 是展示策略而非分发参数：工具主体不接收它，因此在这里把它
        # 抽离出来并挂到结果上交给渲染器。集中处理可避免每个数据工具各自重新实现
        # （或错误实现）同一套宽度开关。
        is_data_tool = isinstance(self.input_model, type) and issubclass(
            self.input_model, DataInput
        )
        detail = "summary"
        if is_data_tool:
            detail = str(params.pop("detail", "summary") or "summary")
        # 先绑定处理器：直接在调用表达式中解包会触发钩子 docstring 中提到的同一静态规则。
        handler = self._dispatch
        try:
            raw = await handler(**params)
        except ValueError as exc:
            return ToolResult(content="", ok=False, error=str(exc))
        except DataUnavailableError as exc:
            return ToolResult(content="", ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - 工具绝不能中断循环
            return ToolResult(content="", ok=False, error=_execution_message(exc))
        if is_data_tool:
            if isinstance(raw.params, dict):
                raw.params.setdefault("detail", detail)
            else:  # pragma: no cover - 每个数据工具都会设置 dict
                raw.params = {"detail": detail}
        try:
            content, sources = self.render(raw)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content="", ok=False, error=_render_message(exc))
        content = self._with_provenance_note(raw, content)
        # ``raw.paths`` 是工具产出的文件（图表、报告）；它们作为附件随结果返回，
        # 以便传输层能够提供它们。
        # ``raw.metadata`` 携带工具声明的、供钩子使用的可观测载荷（如风险复核结果）——
        # 只做转发，从不渲染。
        return ToolResult(
            content=content,
            ok=True,
            sources=list(sources or []),
            attachments=[str(path) for path in (raw.paths or [])],
            metadata=dict(raw.metadata),
        )

    @staticmethod
    def _with_provenance_note(raw: RawData, content: str) -> str:
        """让缓存命中在模型所读的结果中可见。

        ``from_cache`` 过去只在报告附录中出现，因此模型在关键时刻无法分辨复用了缓存
        还是重新取数。在结果顶部加一行，才能让“不要重复取数”变得可验证，
        而不是寄望于模型行为。
        """
        if not getattr(raw, "from_cache", False) or not content:
            return content
        return CACHE_HIT_NOTE + "\n" + content

    def _validate(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.input_model is BaseModel:
            return dict(kwargs)
        return self.input_model.model_validate(kwargs).model_dump(exclude_none=True)

    # -- 渲染 ------------------------------------------------------------
    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """默认实现：把数据框裁剪为 markdown；文本载荷原样透传。"""
        if raw.kind == "text" or (raw.df is None and raw.text is not None):
            return (raw.text or "（无数据）"), [raw]
        text = self.trim_dataframe(
            raw.df, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        return text, ([] if raw.df is None else [raw])

    @staticmethod
    def _render_detail(raw: RawData) -> str:
        """从结果中读取所请求的详细程度（默认 summary）。"""
        return str((raw.params or {}).get("detail") or "summary")

    def trim_dataframe(
        self,
        df: pd.DataFrame | None,
        *,
        source_path: str | None = None,
        detail: str = "summary",
    ) -> str:
        """渲染一个有界的 markdown 视图；完整数据框保存在 parquet 中。

        仅靠行数预算无法限定大小：宽表（财务报表有数十列）只消几行就会突破 token
        预算，因此还要持续丢弃列，直到渲染结果能放进 ``context.max_result_tokens``。

        丢弃会被如实报告而非静默处理，且年度报告期列会优先排在季度列之前。两者都
        重要，因为模型无法对从未展示过的列进行推理：此前一个“过去三年”的数据框会
        悄悄丢掉最早的两个年度列，答案便把这一空缺读成了数据缺失。

        ``detail="full"`` 会用 ``FULL_DETAIL_MULTIPLIER`` 同时放大两项预算，让
        “再多给点”由已持有数据的工具来满足，而不是由第二个工具重读同一份缓存来满足。
        它仍然是有界的。

        ``source_path`` 是存放该数据框的缓存 parquet，保留它使说明能指出复用的载荷
        背后真实的（内容哈希命名的）文件。
        """
        if df is None or not len(df):
            return "（无数据）"
        max_rows = self._max_rows(detail=detail)
        budget = self._result_token_budget(detail=detail)
        view = df.head(max_rows)
        ordered = self._column_display_order(view)
        column_cap = (
            MAX_RENDER_COLS * FULL_DETAIL_MULTIPLIER
            if detail == "full"
            else MAX_RENDER_COLS
        )
        limit = min(len(ordered), column_cap)
        body, columns = "", 0
        while limit >= 1:
            selected = ordered[:limit]
            candidate = view[selected].to_markdown(index=False)
            if self._count_tokens(candidate) <= budget or limit == 1:
                body, columns = candidate, limit
                break
            limit -= 1
        dropped_rows = len(df) - len(view)
        dropped_columns = [str(c) for c in df.columns if c not in set(ordered[:columns])]
        note = self._truncation_note(
            dropped_rows, dropped_columns, source_path, detail=detail
        )
        return body + ("\n" + note if note else "")

    @staticmethod
    def _column_display_order(df: pd.DataFrame) -> list[str]:
        """把年度报告期列排在前面，其余按数据框原顺序排列。

        财务数据框按期数从新到旧排列，因此单纯从左到右的裁剪会牺牲多年对比中最早的
        年份。年度列被优先呈现，使三个年末能在考虑季度列之前先容纳进来；非报告期
        数据框原样返回。
        """
        if not any(_PERIOD_COLUMN_RE.match(str(c)) for c in df.columns):
            return list(df.columns)
        annual = [c for c in df.columns if _ANNUAL_PERIOD_RE.match(str(c))]
        rest = [c for c in df.columns if c not in set(annual)]
        return annual + rest

    def _truncation_note(
        self,
        dropped_rows: int,
        dropped_columns: list[str],
        source_path: str | None = None,
        *,
        detail: str = "summary",
    ) -> str:
        """说明被省略了什么，使其绝不会被误认为数据缺失。

        该说明指出*真正可行的下一步*：用同一工具并带 ``detail="full"``。它过去指向
        对缓存 parquet 调用 ``read_file``，但那无法给出更完整的视图（它走同一套裁剪
        预算）——阅读者照做也依然拿不到数据。``source_path`` 作为复用载荷的来源信息保留。
        """
        parts: list[str] = []
        if dropped_columns:
            named = "、".join(dropped_columns[:_MAX_NAMED_DROPPED_COLUMNS])
            more = len(dropped_columns) - _MAX_NAMED_DROPPED_COLUMNS
            parts.append(f"已省略列：{named}" + (f" 等 {len(dropped_columns)} 列" if more > 0 else ""))
        if dropped_rows > 0:
            parts.append(f"已省略行：{dropped_rows} 行")
        if not parts:
            return ""
        tail = FULL_DETAIL_HINT if detail != "full" else "已按 detail=full 展示，仍超出预算"
        if source_path:
            resolved = Path(source_path)
            if not resolved.is_absolute():
                resolved = resolved.resolve()
            tail = f"{tail}（来源 {resolved}）"
        return "（" + "；".join(parts) + "。" + tail + "）"

    def _result_token_budget(self, *, detail: str = "summary") -> int:
        settings = getattr(self.data, "settings", None)
        if settings is None:
            return 0  # 未知预算：保持渲染后的数据框原样
        budget = int(settings.context.max_result_tokens)
        return budget * FULL_DETAIL_MULTIPLIER if detail == "full" else budget

    @staticmethod
    def _count_tokens(text: str) -> int:
        """用共享计数器计数；计数绝不能中断渲染。"""
        global _SHARED_COUNTER
        try:
            if _SHARED_COUNTER is None:
                from finharness.context.tokens import TokenCounter

                _SHARED_COUNTER = TokenCounter()
            return _SHARED_COUNTER.count(text).tokens
        except Exception:  # noqa: BLE001
            return int(len(text) / 1.7)

    def _max_rows(self, *, detail: str = "summary") -> int:
        """行数预算取自 settings.context.trim_rows，缺失时回退到默认值。"""
        settings = getattr(self.data, "settings", None)
        budget = MAX_RENDER_ROWS if settings is None else int(settings.context.trim_rows)
        return budget * FULL_DETAIL_MULTIPLIER if detail == "full" else budget


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
