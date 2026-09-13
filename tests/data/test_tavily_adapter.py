"""Tavily adapter: request shaping, response parsing, failure isolation.

Uses httpx's MockTransport so nothing here touches the network — the adapter is
the boundary where a bad payload would otherwise become a confusing crash
somewhere downstream.
"""

from __future__ import annotations

import json

import httpx
import pytest

from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.tavily_adapter import (
    EXTRACT_INTERFACE,
    SEARCH_INTERFACE,
    TavilyAdapter,
    validate_web_url,
)


def make_adapter(handler, *, api_key="k"):
    return TavilyAdapter(
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def sent_payload(request: httpx.Request) -> dict:
    """The JSON body the adapter sent, parsed rather than string-matched."""
    return json.loads(request.read().decode())


# -- search -------------------------------------------------------------------


def test_search_parses_results_into_a_title_url_content_frame():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "茅台公告", "url": "https://x.com/a", "content": "摘要 A"},
                    {"title": "行业新闻", "url": "https://y.com/b", "content": "摘要 B"},
                ]
            },
        )

    adapter = make_adapter(handler)
    result = adapter.fetch_web_search("贵州茅台", 5)

    assert result.interface == SEARCH_INTERFACE
    assert list(result.df.columns) == ["title", "url", "content"]
    assert result.df.iloc[0]["title"] == "茅台公告"
    assert result.df.iloc[1]["url"] == "https://y.com/b"
    assert captured["url"] == "https://api.tavily.com/search"
    # The key travels as a bearer header, not as a query parameter.
    assert captured["auth"] == "Bearer k"


def test_search_sends_bearer_auth_and_optional_filters():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = sent_payload(request)
        return httpx.Response(200, json={"results": []})

    adapter = make_adapter(handler)
    adapter.fetch_web_search("货币政策", 3, topic="news", time_range="week")

    assert seen["auth"] == "Bearer k"
    assert seen["payload"]["topic"] == "news"
    assert seen["payload"]["time_range"] == "week"


def test_search_omits_filters_when_not_requested():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = sent_payload(request)
        return httpx.Response(200, json={"results": []})

    adapter = make_adapter(handler)
    adapter.fetch_web_search("茅台", 5)

    assert "topic" not in seen["payload"]
    assert "time_range" not in seen["payload"]
    # Full pages and model-authored answers are deliberately never requested.
    assert seen["payload"]["include_answer"] is False
    assert seen["payload"]["include_raw_content"] is False


def test_search_clamps_top_n_to_the_api_range():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = sent_payload(request)
        return httpx.Response(200, json={"results": []})

    make_adapter(handler).fetch_web_search("茅台", 99)

    assert seen["payload"]["max_results"] == 20


def test_search_tolerates_a_missing_results_key():
    adapter = make_adapter(lambda request: httpx.Response(200, json={}))

    result = adapter.fetch_web_search("无结果的关键词", 5)

    assert len(result.df) == 0
    assert list(result.df.columns) == ["title", "url", "content"]


# -- fetch --------------------------------------------------------------------


def test_fetch_url_returns_raw_content():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = sent_payload(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://x.com/a", "raw_content": "# 正文\n内容"},
                ],
                "failed_results": [],
            },
        )

    adapter = make_adapter(handler)
    result = adapter.fetch_url("https://x.com/a", query="主营构成")

    assert result.interface == EXTRACT_INTERFACE
    assert result.df.iloc[0]["content"] == "# 正文\n内容"
    assert captured["url"] == "https://api.tavily.com/extract"
    assert captured["payload"]["format"] == "markdown"
    assert captured["payload"]["query"] == "主营构成"


def test_fetch_url_reports_the_failed_url_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [],
                "failed_results": [{"url": "https://x.com/a", "error": "404 Not Found"}],
            },
        )

    adapter = make_adapter(handler)
    with pytest.raises(AdapterError) as exc:
        adapter.fetch_url("https://x.com/a")

    assert "404 Not Found" in str(exc.value)


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://host/x", "not-a-url", "javascript:alert(1)"],
)
def test_non_http_urls_are_refused_before_any_request(url):
    called = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request)
        return httpx.Response(200, json={"results": []})

    adapter = make_adapter(handler)
    with pytest.raises(ValueError):
        adapter.fetch_url(url)

    assert called == []


def test_validate_web_url_accepts_http_and_https():
    assert validate_web_url("https://example.com/a?b=1") == "https://example.com/a?b=1"
    assert validate_web_url("http://example.com") == "http://example.com"


# -- failures -----------------------------------------------------------------


def test_missing_key_is_reported_as_a_clear_adapter_error():
    adapter = TavilyAdapter(api_key=None)

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_web_search("茅台", 5)

    assert "未配置" in str(exc.value)
    assert not exc.value.retryable


@pytest.mark.parametrize(
    "status,retryable",
    [(401, False), (433, False), (429, True), (500, True)],
)
def test_http_errors_map_to_retryable_flags(status, retryable):
    adapter = make_adapter(
        lambda request: httpx.Response(status, json={"detail": {"error": "boom"}})
    )

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_web_search("茅台", 5)

    assert exc.value.retryable is retryable
    assert "boom" in str(exc.value)


def test_network_failure_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(AdapterError) as exc:
        make_adapter(handler).fetch_web_search("茅台", 5)

    assert exc.value.retryable is True


def test_non_json_response_is_an_adapter_error():
    adapter = make_adapter(lambda request: httpx.Response(200, text="<html>oops</html>"))

    with pytest.raises(AdapterError):
        adapter.fetch_web_search("茅台", 5)


def test_unsupported_topic_is_refused_without_a_request():
    """topic=finance is broken for Chinese queries, so it is rejected up front."""
    called = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request)
        return httpx.Response(200, json={"results": []})

    adapter = make_adapter(handler)

    for topic in ("finance", "science", "whatever"):
        with pytest.raises(ValueError):
            adapter.fetch_web_search("贵州茅台 营收", 5, topic=topic)

    assert called == []


def test_supported_topics_are_sent_through():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = sent_payload(request)
        return httpx.Response(200, json={"results": []})

    make_adapter(handler).fetch_web_search("贵州茅台", 5, topic="news")

    assert seen["payload"]["topic"] == "news"
