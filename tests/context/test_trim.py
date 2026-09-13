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


def test_dropped_columns_are_named_not_silently_omitted(tmp_path):
    """A reader cannot reason about a column that was never shown."""
    import pandas as pd

    settings = Settings(
        context=ContextSettings(max_result_tokens=1000),
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = _StubTool(DataAccess([], settings=settings))
    # Wide frame: the token budget forces columns to be dropped.
    df = pd.DataFrame({f"col_{i}": range(40) for i in range(12)})

    rendered = tool.trim_dataframe(df)

    assert "已省略列" in rendered
    assert "parquet" in rendered


def test_annual_period_columns_are_ordered_ahead_of_quarters():
    """Column trimming is left-to-right, so annuals must lead to survive."""
    import pandas as pd

    from finharness.tools.base import BaseTool

    df = pd.DataFrame(
        {c: [1.0] for c in ("选项", "指标", "20260630", "20251231", "20241231", "20231231")}
    )

    ordered = BaseTool._column_display_order(df)

    # Three year-ends first (newest first), then the newest quarter and labels.
    assert ordered[:3] == ["20251231", "20241231", "20231231"]
    assert ordered[3:] == ["选项", "指标", "20260630"]


def test_trimming_a_wide_period_frame_keeps_the_year_ends(tmp_path):
    import pandas as pd

    settings = Settings(
        context=ContextSettings(max_result_tokens=300, trim_rows=20),
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = _StubTool(DataAccess([], settings=settings))
    df = pd.DataFrame(
        {c: list(range(20)) for c in
         ("选项", "指标", "20260630", "20260331", "20251231", "20250930", "20250630",
          "20241231", "20240930", "20231231")}
    )

    rendered = tool.trim_dataframe(df)
    header = rendered.splitlines()[0]

    # The three year-ends survive trimming; quarterly columns are sacrificed.
    assert "20231231" in header and "20241231" in header and "20251231" in header
    assert "已省略列" in rendered


def test_non_period_frames_keep_their_original_column_order(tmp_path):
    import pandas as pd

    settings = Settings(
        context=ContextSettings(max_result_tokens=1000),
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = _StubTool(DataAccess([], settings=settings))
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "close": [1.0, 2.0, 3.0]})

    rendered = tool.trim_dataframe(df)

    assert rendered.splitlines()[0].startswith("| date")
    assert "已省略列" not in rendered


class _StubTool(BaseTool):
    name = "stub"
    description = "stub"

    async def _dispatch(self, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError
