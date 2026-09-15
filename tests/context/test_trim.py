"""上下文裁剪：schema description 与 frame 行都受预算约束。"""

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
        # 极小的预算会以 20 个字符为下限，以保证 description 可读；
        # 已经更短的内容则保持不变。
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

    # 表头 + 分隔行 + 恰好 trim_rows 行数据，外加截断提示。
    assert "|   0 |" in rendered
    assert "|   2 |" in rendered
    assert "|   3 |" not in rendered


def test_dropped_columns_are_named_not_silently_omitted(tmp_path):
    """读者无法对从未展示过的列进行推断。"""
    import pandas as pd

    settings = Settings(
        context=ContextSettings(max_result_tokens=1000),
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = _StubTool(DataAccess([], settings=settings))
    # 宽 frame：token 预算迫使部分列被丢弃。
    df = pd.DataFrame({f"col_{i}": range(40) for i in range(12)})

    rendered = tool.trim_dataframe(df)

    assert "已省略列" in rendered
    # 提示会指明*真正有效的下一步*：以 detail=full 调用本工具。
    assert 'detail="full"' in rendered


def test_truncation_note_points_at_detail_full_not_read_file(tmp_path):
    """提示必须指明一个确实能提供更多内容的步骤。

    它曾经写作“用 read_file 读取缓存的 parquet”，但那同样要经过 trim
    预算，因此返回的内容并不比原先的取数更多。真正的下一步是以
    detail="full" 调用同一个工具。
    """
    import pandas as pd

    settings = Settings(
        context=ContextSettings(max_result_tokens=1000),
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = _StubTool(DataAccess([], settings=settings))
    source = tmp_path / "cache" / "parquet" / "2026-09" / "abc123.parquet"
    df = pd.DataFrame({"v": range(50)})

    rendered = tool.trim_dataframe(df, source_path=str(source))

    assert 'detail="full"' in rendered
    assert "read_file" not in rendered
    # 复用载荷的来源仍然会被标明。
    assert "abc123.parquet" in rendered


def test_truncation_note_path_is_absolute(tmp_path, monkeypatch):
    """相对路径会使结果取决于模型能否猜中工作目录。"""
    import pandas as pd

    settings = Settings(
        context=ContextSettings(max_result_tokens=1000),
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = _StubTool(DataAccess([], settings=settings))
    df = pd.DataFrame({"v": range(50)})

    rendered = tool.trim_dataframe(df, source_path="data_cache/parquet/x.parquet")

    assert "来源 " in rendered
    named = rendered.split("来源 ")[1].split("）")[0]
    from pathlib import Path

    assert Path(named).is_absolute()


def test_note_still_names_detail_full_without_a_source_path(tmp_path):
    import pandas as pd

    settings = Settings(
        context=ContextSettings(max_result_tokens=1000),
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = _StubTool(DataAccess([], settings=settings))

    rendered = tool.trim_dataframe(pd.DataFrame({"v": range(50)}))

    assert 'detail="full"' in rendered


def test_full_detail_widens_the_row_budget(tmp_path):
    """detail="full" 是同一个工具的一种策略：它必须返回比 summary
    更多的行，而不是完全相同的切片。"""
    import pandas as pd

    settings = Settings(
        context=ContextSettings(trim_rows=5, max_result_tokens=100000),
        data={"cache_dir": tmp_path / "cache"},
    )
    tool = _StubTool(DataAccess([], settings=settings))
    df = pd.DataFrame({"v": range(100)})

    def data_rows(text: str) -> int:
        lines = text.splitlines()
        # 去掉 markdown 表头（第 0 行）和 `|---|` 分隔行（第 1 行）。
        return sum(1 for line in lines[2:] if line.startswith("|"))

    summary = tool.trim_dataframe(df, detail="summary")
    full = tool.trim_dataframe(df, detail="full")

    assert data_rows(summary) == 5
    assert data_rows(full) > 5


def test_annual_period_columns_are_ordered_ahead_of_quarters():
    """列裁剪是从左到右进行的，因此年度列必须排在前面才能被保留。"""
    import pandas as pd

    from finharness.tools.base import BaseTool

    df = pd.DataFrame(
        {c: [1.0] for c in ("选项", "指标", "20260630", "20251231", "20241231", "20231231")}
    )

    ordered = BaseTool._column_display_order(df)

    # 先放三个年末（最新的在前），然后是最近的季度和标签列。
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

    # 三个年末在裁剪中保留下来；季度列则被舍弃。
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

    async def _dispatch(self, **kwargs):  # pragma: no cover - 未被调用
        raise NotImplementedError
