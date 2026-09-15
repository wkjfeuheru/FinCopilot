"""persist_frame：派生的 frame 会获得可复用的 parquet 路径，供 citation 使用。"""

import pandas as pd

from finharness.data.frames import persist_frame


def frame(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=rows), "nav": range(rows)})


def test_frame_is_written_and_path_returned(tmp_path):
    path = persist_frame(frame(), cache_dir=tmp_path, name="backtest_600519")

    assert path is not None
    assert path.endswith(".parquet")
    loaded = pd.read_parquet(path)
    assert list(loaded.columns) == ["date", "nav"]


def test_same_frame_reuses_the_same_file(tmp_path):
    first = persist_frame(frame(), cache_dir=tmp_path, name="bt")
    second = persist_frame(frame(), cache_dir=tmp_path, name="bt")

    assert first == second


def test_different_frames_get_different_files(tmp_path):
    first = persist_frame(frame(3), cache_dir=tmp_path, name="bt")
    second = persist_frame(frame(6), cache_dir=tmp_path, name="bt")

    assert first != second


def test_empty_frame_is_not_written(tmp_path):
    assert persist_frame(pd.DataFrame(), cache_dir=tmp_path, name="bt") is None


def test_unwritable_location_degrades_to_none(tmp_path):
    """持久化只是一种优化：失败时不得抛出异常。"""
    blocker = tmp_path / "frames"
    blocker.write_text("not a directory", encoding="utf-8")

    assert persist_frame(frame(), cache_dir=tmp_path, name="bt") is None
