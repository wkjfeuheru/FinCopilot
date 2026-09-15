"""工具原始数据的统一载体（docs 03.4.2）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RawData:
    """工具的 ``execute`` 在渲染之前返回的内容。

    ``endpoint`` 标明实际提供数据的接口（例如
    ``akshare:stock_zh_a_hist``），从而使引用与缓存键在适配器
    回退过程中保持准确。
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
    # 工具声明的可观测性载荷（例如审查结果），随结果一同携带
    # 供钩子使用。从不渲染，也从不发送给服务提供方。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return int(len(self.df)) if self.df is not None else 0
