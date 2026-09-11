"""Small M0 registry for the first financial tools."""

from finharness.tools.fin.indicators import GetIndicatorsTool
from finharness.tools.fin.kline import GetKlineTool
from finharness.tools.fin.quote import GetQuoteTool


class ToolRegistry:
    def __init__(self, data):
        self.tools = {
            tool.name: tool(data)
            for tool in (GetQuoteTool, GetKlineTool, GetIndicatorsTool)
        }

    def names(self) -> list[str]:
        return list(self.tools)

    def resolve(self, name: str):
        return self.tools.get(name)

    def schemas(self, names: set[str] | None = None) -> list[dict]:
        parameters = {
            "get_quote": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
            "get_kline": {"type": "object", "properties": {"symbol": {"type": "string"}, "period": {"type": "string"}, "adjust": {"type": ["string", "null"]}, "years": {"type": "integer"}}, "required": ["symbol"]},
            "get_indicators": {"type": "object", "properties": {"symbol": {"type": "string"}, "years": {"type": "integer"}, "fields": {"type": ["array", "null"], "items": {"type": "string"}}}, "required": ["symbol"]},
        }
        return [
            {"type": "function", "function": {"name": name, "description": tool.description, "parameters": parameters[name]}}
            for name, tool in self.tools.items()
            if names is None or name in names
        ]

    def is_read_only(self, name: str) -> bool:
        return name in {"get_quote", "get_kline", "get_indicators"}
