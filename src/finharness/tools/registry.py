"""Two-tier tool registry: resident schemas plus searchable lazy tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from finharness.context.trim import DEFAULT_MAX_DESC_LEN, trim_schema
from finharness.data.access import DataAccess
from finharness.tools.base import BaseTool, PermissionLevel

# Tool classes are registered through a loader so the module stays importable
# while the catalogue grows.
from finharness.tools.fin.announcements import GetAnnouncementsTool
from finharness.tools.fin.chart import MakeChartTool
from finharness.tools.fin.financials import GetFinancialsTool
from finharness.tools.fin.indicators import GetIndicatorsTool
from finharness.tools.fin.kline import GetKlineTool
from finharness.tools.fin.metrics import CalcMetricsTool
from finharness.tools.fin.news import GetMarketNewsTool
from finharness.tools.fin.peers import GetPeersTool
from finharness.tools.fin.quote import GetQuoteTool
from finharness.tools.fin.valuation import GetValuationTool
from finharness.tools.fin.valuation_calc import CalcValuationTool
from finharness.tools.fin.writer import WriteReportTool
from finharness.tools.generic.files import ReadFileTool, WriteFileTool
from finharness.tools.meta.ask import AskUserTool
from finharness.tools.meta.discovery import LoadToolTool, SearchToolsTool
from finharness.tools.meta.plan import ResearchPlanTool
from finharness.tools.meta.skills import ListSkillsTool, LoadSkillTool

# docs 03.4.1 权威口径：resident=20、lazy=4。Every name here must resolve to a
# registered class — a lazy entry without an implementation would be listed by
# lazy_names() while being unreachable, so run_backtest / read_pdf are omitted
# until their tools exist.
DEFAULT_LAZY_TOOLS: tuple[str, ...] = (
    "get_announcements",
    "calc_valuation",
)

ALL_TOOL_CLASSES: tuple[type[BaseTool], ...] = (
    # 金融-数据
    GetQuoteTool,
    GetKlineTool,
    GetIndicatorsTool,
    GetFinancialsTool,
    GetValuationTool,
    GetPeersTool,
    GetMarketNewsTool,
    GetAnnouncementsTool,
    # 金融-计算
    CalcMetricsTool,
    CalcValuationTool,
    # 金融-输出
    MakeChartTool,
    WriteReportTool,
    # 通用
    ReadFileTool,
    WriteFileTool,
    # 元
    ResearchPlanTool,
    SearchToolsTool,
    ListSkillsTool,
    LoadSkillTool,
    LoadToolTool,
    AskUserTool,
)

FINANCIAL_DATA_TOOLS = ALL_TOOL_CLASSES


@dataclass(frozen=True, slots=True)
class ToolBrief:
    """Search result: enough to describe a lazy tool without its full schema."""

    name: str
    description: str
    group: str
    score: int = 0


def build_parameters(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for the model, with the title noise stripped."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


class ToolRegistry:
    """Instantiates every tool but only exposes resident schemas by default."""

    def __init__(self, data: DataAccess, *, ctx: Any | None = None, settings: Any | None = None) -> None:
        self.settings = settings
        self.tools: dict[str, BaseTool] = {
            tool_cls.name: tool_cls(data, ctx=ctx) for tool_cls in ALL_TOOL_CLASSES
        }
        # Meta tools need the catalogue they live in; wire after construction.
        for tool in self.tools.values():
            tool.registry = self
        configured_lazy = tuple(getattr(settings.tools, "lazy", ()) or ()) if settings else ()
        configured_resident = tuple(getattr(settings.tools, "resident", ()) or ()) if settings else ()
        if configured_resident:
            # An explicit resident list wins; everything else becomes lazy.
            self._lazy = {name for name in self.tools if name not in configured_resident}
        elif configured_lazy:
            self._lazy = {name for name in configured_lazy if name in self.tools}
        else:
            self._lazy = {name for name in DEFAULT_LAZY_TOOLS if name in self.tools}
        # Lazy tools only enter the request after activation (docs 03.4.3).
        self._active: set[str] = {name for name in self.tools if name not in self._lazy}

    # -- catalogue ------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self.tools)

    def resident_names(self) -> list[str]:
        return [name for name in self.tools if name not in self._lazy]

    def lazy_names(self) -> list[str]:
        return [name for name in self.tools if name in self._lazy]

    def resolve(self, name: str) -> BaseTool | None:
        return self.tools.get(name)

    def is_read_only(self, name: str) -> bool:
        tool = self.tools.get(name)
        return tool is not None and tool.permission is PermissionLevel.READ

    def is_active(self, name: str) -> bool:
        return name in self._active

    def activate(self, name: str) -> bool:
        """Add a tool to the injected set; returns False for unknown or already-active."""
        if name not in self.tools or name in self._active:
            return False
        self._active.add(name)
        return True

    # -- schemas --------------------------------------------------------------
    def schemas(self, names: set[str] | None = None) -> list[dict]:
        """OpenAI-style schemas for the injected set (registry order).

        ``names`` defaults to the active set, so lazy tools stay hidden until
        the round after activation. Descriptions are trimmed to the configured
        budget because every resident schema is re-sent on every request
        (docs 3.6.1).
        """
        selected = self._active if names is None else names
        budget = self._desc_budget()
        return [
            trim_schema(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": self._describe(tool),
                        "parameters": build_parameters(tool.input_model),
                    },
                },
                max_desc_len=budget,
            )
            for name, tool in self.tools.items()
            if name in selected
        ]

    def _desc_budget(self) -> int:
        """Per-description character budget derived from the token setting."""
        if self.settings is None:
            return DEFAULT_MAX_DESC_LEN
        # Tokens -> characters using the documented Chinese approximation, so the
        # setting stays expressed in the unit the docs use.
        return max(int(self.settings.context.max_tool_schema_tokens * 1.7), 20)

    # -- search ---------------------------------------------------------------
    def search(self, query: str, *, limit: int = 5) -> list[ToolBrief]:
        """Keyword-weighted search over name/description, cataloguing lazy tools.

        v1 is deliberately word-matching plus recency-free weighting (docs
        3.4.3): no vector store.
        """
        terms = [term for term in query.lower().replace("，", " ").split() if term]
        scored: list[ToolBrief] = []
        for name, tool in self.tools.items():
            haystack = f"{name} {tool.description}".lower()
            score = 0
            for term in terms:
                if term in name.lower():
                    score += 3
                if term in haystack:
                    score += 1
            if score:
                scored.append(
                    ToolBrief(
                        name=name,
                        description=tool.description,
                        group=tool.group.value,
                        score=score,
                    )
                )
        scored.sort(key=lambda brief: (-brief.score, brief.name))
        return scored[:limit]

    @staticmethod
    def _describe(tool: BaseTool) -> str:
        if tool.output_schema_note:
            return tool.description + " " + tool.output_schema_note
        return tool.description
