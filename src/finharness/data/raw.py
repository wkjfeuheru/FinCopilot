"""Unified carrier for raw tool data (docs 03.4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RawData:
    """What a tool's ``execute`` returns before rendering.

    ``endpoint`` names the interface that actually served the data (for
    example ``akshare:stock_zh_a_hist``) so citations and cache keys stay
    accurate across adapter fallback.
    """

    kind: str = "df"  # "df" | "chart_path" | "text" | "pdf_text"
    df: Any = None
    text: str | None = None
    paths: list[str] = field(default_factory=list)
    endpoint: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    data_date: str | None = None
    from_cache: bool = False
    cache_key: str | None = None
    parquet_path: str | None = None

    @property
    def rows(self) -> int:
        return int(len(self.df)) if self.df is not None else 0
