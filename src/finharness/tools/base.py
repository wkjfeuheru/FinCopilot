"""Base financial tool contract."""

from finharness.data.access import DataAccess, DataUnavailableError
from finharness.types import ToolResult


class BaseTool:
    name = "tool"
    description = ""

    def __init__(self, data: DataAccess):
        self.data = data

    async def run(self, **kwargs) -> ToolResult:
        try:
            return await self.execute(**kwargs)
        except (ValueError, DataUnavailableError) as exc:
            return ToolResult(content="", ok=False, error=str(exc))
        except Exception as exc:
            return ToolResult(content="", ok=False, error=f"tool execution failed: {exc}")

    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    @staticmethod
    def render_dataframe(dataframe) -> str:
        return dataframe.head(20).to_markdown(index=False)
