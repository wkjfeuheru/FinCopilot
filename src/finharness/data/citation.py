"""Citation registry: every rendered datum is traceable to its source."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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
    """Numeric part of a ``cit_%06d`` id; 0 when the shape is unexpected."""
    _, _, suffix = cid.partition("_")
    return int(suffix) if suffix.isdigit() else 0


def fingerprint_series(records: Any) -> str:
    """Stable digest of tabular content, independent of column order noise."""
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_frame(df: Any) -> str:
    """Digest a DataFrame (or None) for citation provenance."""
    if df is None or not len(df):
        return fingerprint_series([])
    return fingerprint_series(df.head(50).to_dict("records"))


def fingerprint_text(text: str | None) -> str:
    """Digest a text payload (a page body, a file read) for provenance.

    Without this, every text-only citation shared one constant fingerprint —
    the digest of an empty frame — so two different documents were
    indistinguishable in the appendix.
    """
    if not text:
        return fingerprint_series([])
    return fingerprint_series({"text": text})


class CitationRegistry:
    """Session-scoped citation store; bounded to avoid unbounded growth."""

    def __init__(self, *, max_entries: int = 500) -> None:
        self._max_entries = max_entries
        self._items: list[Citation] = []
        self._by_id: dict[str, Citation] = {}
        self._counter = 0

    def restore(self, citations: list[Citation]) -> None:
        """Reinstate previously issued citations, keeping their original ids.

        Id stability matters once conversations persist: a stored summary or
        conclusion names ``cit_000005``, so renumbering on reload would silently
        point those references at different data.
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
