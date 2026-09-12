"""Two-tier tool registry: schemas generated from each tool's input model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from finharness.data.access import DataAccess
from finharness.tools.base import BaseTool, PermissionLevel
from finharness.tools.fin.announcements import GetAnnouncementsTool
from finharness.tools.fin.financials import GetFinancialsTool
from finharness.tools.fin.indicators import GetIndicatorsTool
from finharness.tools.fin.kline import GetKlineTool
from finharness.tools.fin.news import GetMarketNewsTool
from finharness.tools.fin.peers import GetPeersTool
from finharness.tools.fin.quote import GetQuoteTool
from finharness.tools.fin.valuation import GetValuationTool

FINANCIAL_DATA_TOOLS: tuple[type[BaseTool], ...] = (
    GetQuoteTool,
    GetKlineTool,
    GetIndicatorsTool,
    GetFinancialsTool,
    GetValuationTool,
    GetPeersTool,
    GetMarketNewsTool,
    GetAnnouncementsTool,
)


def build_parameters(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for the model, with the title noise stripped."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


class ToolRegistry:
    """Instantiates the read-only financial tools and derives their schemas."""

    def __init__(self, data: DataAccess) -> None:
        self.tools: dict[str, BaseTool] = {
            tool_cls.name: tool_cls(data) for tool_cls in FINANCIAL_DATA_TOOLS
        }

    def names(self) -> list[str]:
        return list(self.tools)

    def resolve(self, name: str) -> BaseTool | None:
        return self.tools.get(name)

    def is_read_only(self, name: str) -> bool:
        tool = self.tools.get(name)
        return tool is not None and tool.permission is PermissionLevel.READ

    def schemas(self, names: set[str] | None = None) -> list[dict]:
        """OpenAI-style tool schemas, in registry order."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self._describe(tool),
                    "parameters": build_parameters(tool.input_model),
                },
            }
            for name, tool in self.tools.items()
            if names is None or name in names
        ]

    @staticmethod
    def _describe(tool: BaseTool) -> str:
        if tool.output_schema_note:
            return tool.description + " " + tool.output_schema_note
        return tool.description
