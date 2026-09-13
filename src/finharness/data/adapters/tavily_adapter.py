"""Tavily adapter: external web search and URL content extraction (docs 03.4).

Tavily exposes two endpoints, and using both here has a security consequence
worth stating plainly: **this process never fetches an arbitrary user-supplied
URL itself**. For ``fetch_url`` the outbound request is made by Tavily's
servers, so there is no SSRF surface in this codebase — no private-range
blocking, no redirect re-validation. That is the whole reason to prefer the
hosted extractor over a local HTTP client.

The trade-off is capability: intranet and local addresses cannot be fetched,
and every URL is disclosed to Tavily. Both are acceptable for public financial
research, which is the only use this is meant for.

Results are returned as small DataFrames so they ride the project's existing
cache, citation and adapter-fallback machinery instead of a parallel path.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd

from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult

SEARCH_INTERFACE = "search"
EXTRACT_INTERFACE = "extract"

# Columns the search frame carries. Kept deliberately tight: title and url are
# what a reader cites, content is the snippet the model judges relevance on.
SEARCH_COLUMNS = ("title", "url", "content")
FETCH_COLUMNS = ("url", "content")

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Tavily's ``topic="finance"`` returns badly irrelevant results for Chinese
# queries — verified against the live API, where a Moutai revenue query came
# back with MMA and art-auction pages. Since this project searches in Chinese,
# only the values that actually behave are accepted; ``general`` is the default
# and ``news`` is fine for time-sensitive items.
_ALLOWED_TOPICS = frozenset({"general", "news"})


def validate_web_url(url: str) -> str:
    """Reject anything that is not an absolute http(s) URL.

    This is not SSRF defence — Tavily makes the request, not us. It is input
    hygiene: it stops ``file://``, ``ftp://`` and relative paths from being sent
    to the provider as if they were fetchable pages.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise ValueError(f"只支持 http/https 绝对网址：{url}")
    return url.strip()


def _message_from(response: httpx.Response) -> str:
    """Tavily carries its error text under ``detail.error``."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    detail = payload.get("detail")
    if isinstance(detail, dict) and detail.get("error"):
        return str(detail["error"])
    if isinstance(detail, str):
        return detail
    return str(payload)[:200]


class TavilyAdapter(DataAdapter):
    """Search and extract via the Tavily API; one key serves both."""

    name = "tavily"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.tavily.com",
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # Injectable so tests exercise parsing without touching the network.
        self._client = client

    # -- transport ------------------------------------------------------------
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise AdapterError(
                "未配置联网检索密钥（TAVILY_API_KEY）：web_search / fetch_url 当前不可用"
            )
        headers = {"Authorization": "Bearer " + self.api_key}
        body = {**payload, "api_key": self.api_key}
        try:
            if self._client is not None:
                response = self._client.post(
                    self.base_url + path, json=body, headers=headers, timeout=self.timeout_s
                )
            else:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(
                        self.base_url + path, json=body, headers=headers
                    )
        except httpx.HTTPError as exc:
            raise AdapterError(f"联网检索请求失败：{exc}", retryable=True) from exc

        if response.status_code >= 400:
            # 429 and 5xx are worth another attempt; auth and quota errors are not.
            retryable = response.status_code == 429 or response.status_code >= 500
            raise AdapterError(
                f"联网检索返回 {response.status_code}：{_message_from(response)}",
                retryable=retryable,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise AdapterError("联网检索返回了非 JSON 响应") from exc

    # -- semantic fetches -----------------------------------------------------
    def fetch_web_search(
        self,
        query: str,
        top_n: int,
        topic: str | None = None,
        time_range: str | None = None,
    ) -> FetchResult:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(int(top_n), 20)),
            "search_depth": "basic",
            # Snippets only: full pages would let the search engine dictate the
            # context budget. Reach for fetch_url when a page is actually needed.
            "include_raw_content": False,
            "include_answer": False,
        }
        if topic:
            if topic not in _ALLOWED_TOPICS:
                # Refusing is the honest failure: sending it through would return
                # plausible-looking but unrelated pages the model might cite.
                raise ValueError(
                    f"不支持的检索领域：{topic}（仅支持 {'、'.join(sorted(_ALLOWED_TOPICS))}）"
                )
            payload["topic"] = topic
        if time_range:
            payload["time_range"] = time_range

        data = self._post("/search", payload)
        rows = [
            {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "content": str(item.get("content") or ""),
            }
            for item in data.get("results") or []
        ]
        frame = pd.DataFrame(rows, columns=list(SEARCH_COLUMNS))
        return FetchResult(df=frame, interface=SEARCH_INTERFACE)

    def fetch_url(self, url: str, query: str | None = None) -> FetchResult:
        target = validate_web_url(url)
        payload: dict[str, Any] = {"urls": [target], "format": "markdown"}
        if query:
            # Reranks extracted chunks against the question, so a long page is
            # trimmed toward what was actually asked.
            payload["query"] = query

        data = self._post("/extract", payload)
        results = data.get("results") or []
        failed = data.get("failed_results") or []
        if not results:
            reason = ""
            if failed:
                reason = "：" + str(failed[0].get("error") or "抓取失败")
            raise AdapterError(f"未能抓取该网页{reason}")

        item = results[0]
        frame = pd.DataFrame(
            [{"url": str(item.get("url") or target), "content": str(item.get("raw_content") or "")}],
            columns=list(FETCH_COLUMNS),
        )
        return FetchResult(df=frame, interface=EXTRACT_INTERFACE)
