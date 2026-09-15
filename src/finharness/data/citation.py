"""引用登记表：每个被渲染的数据点都可追溯到其来源。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Citation:
    cid: str
    tool: str
    endpoint: str
    symbol: str | None
    params: dict[str, Any]
    ts: str
    rows: int
    cols: int
    fingerprint: str
    from_cache: bool = False
    parquet_path: str | None = None

    def to_markdown(self) -> str:
        source = "缓存" if self.from_cache else "实时"
        return (
            f"- 【{self.cid}】工具 `{self.tool}` · 接口 `{self.endpoint}` · "
            f"标的 {self.symbol or '-'} · {self.rows}行/{self.cols}列 · "
            f"来源 {source} · {self.ts} · 指纹 `{self.fingerprint[:12]}`"
        )


def _cid_ordinal(cid: str) -> int:
    """``cit_%06d`` 编号中的数字部分；形状不符合预期时返回 0。"""
    _, _, suffix = cid.partition("_")
    return int(suffix) if suffix.isdigit() else 0


def fingerprint_series(records: Any) -> str:
    """表格内容的稳定摘要，不受列顺序噪声影响。"""
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_frame(df: Any) -> str:
    """为引用溯源计算 DataFrame（或 None）的摘要。"""
    if df is None or not len(df):
        return fingerprint_series([])
    return fingerprint_series(df.head(50).to_dict("records"))


def fingerprint_text(text: str | None) -> str:
    """为溯源计算文本载荷（页面正文、文件读取内容）的摘要。

    若没有它，每个纯文本引用都会共享同一个常量指纹 ——
    即空数据框的摘要 —— 因此两个不同的文档
    在附录中将无法区分。
    """
    if not text:
        return fingerprint_series([])
    return fingerprint_series({"text": text})


class CitationRegistry:
    """会话级引用存储；设有上限以避免无界增长。"""

    def __init__(self, *, max_entries: int = 500) -> None:
        self._max_entries = max_entries
        self._items: list[Citation] = []
        self._by_id: dict[str, Citation] = {}
        self._counter = 0

    def restore(self, citations: list[Citation]) -> None:
        """恢复此前已发出的引用，并保留其原始编号。

        一旦对话需要持久化，编号稳定性就很重要：已存储的摘要或
        结论会以 ``cit_000005`` 指代某条数据，若在重新加载时重新编号，
        就会在不知不觉中把这些引用指向不同的数据。
        """
        for citation in citations:
            if citation.cid in self._by_id:
                continue
            self._items.append(citation)
            self._by_id[citation.cid] = citation
            self._counter = max(self._counter, _cid_ordinal(citation.cid))

    def register(
        self,
        *,
        tool: str,
        endpoint: str,
        symbol: str | None,
        params: dict[str, Any],
        rows: int,
        cols: int,
        fingerprint: str,
        from_cache: bool = False,
        parquet_path: str | None = None,
    ) -> Citation:
        """登记一条新引用并分配递增的 ``cit_%06d`` 编号。

        超过容量上限时丢弃最旧的条目。返回新建的 ``Citation``。
        """
        self._counter += 1
        citation = Citation(
            cid=f"cit_{self._counter:06d}",
            tool=tool,
            endpoint=endpoint,
            symbol=symbol,
            params=dict(params),
            ts=datetime.now().astimezone().isoformat(timespec="seconds"),
            rows=rows,
            cols=cols,
            fingerprint=fingerprint,
            from_cache=from_cache,
            parquet_path=parquet_path,
        )
        self._items.append(citation)
        self._by_id[citation.cid] = citation
        if len(self._items) > self._max_entries:
            dropped = self._items.pop(0)
            self._by_id.pop(dropped.cid, None)
        return citation

    def get(self, cid: str) -> Citation | None:
        return self._by_id.get(cid)

    def query(self, *, symbol: str | None = None, tool: str | None = None) -> list[Citation]:
        return [
            item
            for item in self._items
            if (symbol is None or item.symbol == symbol)
            and (tool is None or item.tool == tool)
        ]

    def all(self) -> list[Citation]:
        return list(self._items)

    def resolve_symbols(self) -> list[str]:
        seen: list[str] = []
        for item in self._items:
            if item.symbol and item.symbol not in seen:
                seen.append(item.symbol)
        return seen

    def to_appendix_md(self) -> str:
        if not self._items:
            return "（本次会话暂无数据引用）"
        lines = ["【数据来源】", *(item.to_markdown() for item in self._items)]
        return "\n".join(lines)


class ScopedCitationRegistry:
    """共享登记表之上的一个视图，只记录*它自己*签发的编号。

    子代理共享会话登记表，以使 cid 保持连续，但每个子代理仍需
    准确报告它新增了哪些引用。通过在前后对共享登记表做差异
    来归属是不对的：只要两个子代理同时运行就会出错 ——
    每个都会把对方签发的 cid 算作自己的。在 ``register`` 处
    记录则无论并发如何都是精确的。

    写入直接透传；读取与 ``restore`` 委托出去，因此它对每个使用者
    （``AgentLoop``、``ResearchContext``）都具备 ``CitationRegistry`` 的鸭子类型。
    """

    def __init__(self, shared: CitationRegistry) -> None:
        self._shared = shared
        self._created: list[str] = []

    @property
    def created(self) -> list[str]:
        """本作用域签发的 cid，按创建顺序排列。"""
        return list(self._created)

    def register(self, **kwargs: Any) -> Citation:
        citation = self._shared.register(**kwargs)
        self._created.append(citation.cid)
        return citation

    def __getattr__(self, name: str) -> Any:
        # all/get/query/resolve_symbols/to_appendix_md/restore 都读取
        # 共享登记表，因此委托能让它们天然保持一致。
        return getattr(self._shared, name)
