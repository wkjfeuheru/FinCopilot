"""AkShare adapter: peer tables must stay attributable and un-polluted.

The EM comparison endpoint mixes company rows with two aggregate rows and,
when ``fields`` is used, used to drop 代码/简称 — leaving a frame the caller
could neither identify nor average correctly (docs 03.5.2).
"""

import pandas as pd

from finharness.data.adapters import akshare_adapter
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.mapping import PEER_COMPANY, PEER_ROW_TYPE_COLUMN, PEER_STAT


class FakeAk:
    """Stands in for the akshare module; returns one canned comparison table."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def stock_zh_valuation_comparison_em(self, symbol: str) -> pd.DataFrame:
        return self._frame.copy()


def peer_frame() -> pd.DataFrame:
    """Two issuers plus the aggregate rows exactly as EM returns them."""
    return pd.DataFrame(
        {
            "排名": ["6.0/24", None, None, "1.0"],
            "代码": ["300272", "行业中值", "行业平均", "688169"],
            "简称": ["开能健康", "行业中值", "行业平均", "石头科技"],
            "市盈率-TTM": [117.9, 26.3, 37.4, 18.5],
            "市净率-MRQ": [2.74, 2.79, 3.36, 2.13],
        }
    )


def make_adapter(monkeypatch, frame: pd.DataFrame) -> AkShareAdapter:
    monkeypatch.setattr(akshare_adapter, "_import_akshare", lambda: FakeAk(frame))
    return AkShareAdapter(throttle_seconds=0)


def test_peer_filter_keeps_identity_columns(monkeypatch):
    adapter = make_adapter(monkeypatch, peer_frame())

    df = adapter.fetch_peers("688169", ["市净率", "市盈率"]).df

    # The rows stay attributable even after a field filter.
    assert {"代码", "简称"}.issubset(df.columns)
    assert {"市净率-MRQ", "市盈率-TTM"}.issubset(df.columns)


def test_peer_rows_are_tagged_as_company_or_statistic(monkeypatch):
    adapter = make_adapter(monkeypatch, peer_frame())

    df = adapter.fetch_peers("688169", ["市净率"]).df

    assert PEER_ROW_TYPE_COLUMN in df.columns
    assert set(df[PEER_ROW_TYPE_COLUMN]) == {PEER_COMPANY, PEER_STAT}
    assert (df[PEER_ROW_TYPE_COLUMN] == PEER_STAT).sum() == 2


def test_peer_companies_sort_before_aggregates(monkeypatch):
    adapter = make_adapter(monkeypatch, peer_frame())

    df = adapter.fetch_peers("688169", ["市净率"]).df

    # Aggregates last, so a naive whole-frame mean is visibly wrong.
    types = list(df[PEER_ROW_TYPE_COLUMN])
    assert types == [PEER_COMPANY, PEER_COMPANY, PEER_STAT, PEER_STAT]


def test_peer_without_fields_keeps_every_column_and_tags_rows(monkeypatch):
    adapter = make_adapter(monkeypatch, peer_frame())

    df = adapter.fetch_peers("688169", None).df

    assert "市盈率-TTM" in df.columns
    assert PEER_ROW_TYPE_COLUMN in df.columns
    assert df.iloc[0][PEER_ROW_TYPE_COLUMN] == PEER_COMPANY


def test_peer_table_without_aggregate_rows_is_untouched(monkeypatch):
    """No aggregates means nothing to tag; the frame must pass through intact."""
    plain = pd.DataFrame(
        {
            "代码": ["688169", "603486"],
            "简称": ["石头科技", "科沃斯"],
            "市净率-MRQ": [2.13, 2.96],
        }
    )
    adapter = make_adapter(monkeypatch, plain)

    df = adapter.fetch_peers("688169", ["市净率"]).df

    assert PEER_ROW_TYPE_COLUMN not in df.columns
    assert list(df["代码"]) == ["688169", "603486"]
