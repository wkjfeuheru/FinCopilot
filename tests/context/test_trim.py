"""Context trimming: schema descriptions and frame rows are budgeted."""

from finharness.config.settings import ContextSettings, Settings
from finharness.context.trim import trim_schema
from finharness.data.access import DataAccess
from finharness.tools.base import BaseTool


def schema(description: str) -> dict:
    return {"type": "function", "function": {"name": "t", "description": description, "parameters": {}}}


def test_short_description_is_left_alone():
    trimmed = trim_schema(schema("简短描述"), max_desc_len=60)

    assert trimmed["function"]["description"] == "简短描述"


def test_long_description_is_shortened_and_marked():
    trimmed = trim_schema(schema("很长的描述" * 40), max_desc_len=30)

    description = trimmed["function"]["description"]
    assert len(description) == 30
    assert description.endswith("…")


def test_schema_shape_is_preserved():
    original = schema("描述")
    original["function"]["parameters"] = {"type": "object"}

    trimmed = trim_schema(original, max_desc_len=60)

    assert trimmed["type"] == "function"
    assert trimmed["function"]["parameters"] == {"type": "object"}


def test_missing_function_block_is_tolerated():
    assert trim_schema({"type": "function"}) == {"type": "function"}


def test_registry_trims_descriptions_to_the_configured_budget(tmp_path):
    from finharness.tools.registry import ToolRegistry

    settings = Settings(
        context=ContextSettings(max_tool_schema_tokens=10),
        data={"cache_dir": tmp_path / "cache"},
    )
    registry = ToolRegistry(DataAccess([], settings=settings), settings=settings)

    for entry in registry.schemas():
        description = entry["function"]["description"]
        # A tiny budget floors at 20 characters so descriptions stay readable;
        # anything already shorter is left untouched.
        assert len(description) <= 20


def test_registry_keeps_descriptions_when_budget_is_generous(tmp_path):
    from finharness.tools.registry import ToolRegistry

    settings = Settings(
        context=ContextSettings(max_tool_schema_tokens=500),
        data={"cache_dir": tmp_path / "cache"},
    )
    registry = ToolRegistry(DataAccess([], settings=settings), settings=settings)

    descriptions = [entry["function"]["description"] for entry in registry.schemas()]
    assert any(len(text) > 20 for text in descriptions)
    assert not any(text.endswith("…") for text in descriptions)


def test_trim_rows_from_settings_controls_rendered_rows(tmp_path):
    import pandas as pd

    settings = Settings(
        context=ContextSettings(trim_rows=3), data={"cache_dir": tmp_path / "cache"}
    )
    access = DataAccess([], settings=settings)
    tool = _StubTool(access)

    rendered = tool.trim_dataframe(pd.DataFrame({"v": range(10)}))

    # Header + separator + exactly trim_rows data rows, plus the truncation note.
    assert "|   0 |" in rendered
    assert "|   2 |" in rendered
    assert "|   3 |" not in rendered


class _StubTool(BaseTool):
    name = "stub"
    description = "stub"

    async def _dispatch(self, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError
