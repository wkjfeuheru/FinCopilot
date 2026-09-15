"""Tavily adapter：请求构造、响应解析、故障隔离。

使用 httpx 的 MockTransport，因此这里不会触及网络——adapter 正是这样一个
边界：否则一个坏的 payload 会在下游某处变成令人困惑的崩溃。
"""

from __future__ import annotations

import json

import httpx
import pytest

from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.tavily_adapter import (
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
    """adapter 发送的 JSON body，经解析得到，而非做字符串匹配。"""
    return json.loads(request.read().decode())


# -- 搜索 -------------------------------------------------------------------


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
    # key 以 bearer header 的形式传输，而不是作为查询参数。
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
    # 完整页面和由模型生成的答案都刻意从不请求。
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


# -- url 筛查（与本地 PDF 抓取共用） --------------------------


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://host/x", "not-a-url", "javascript:alert(1)"],
)
def test_non_http_urls_are_refused(url):
    with pytest.raises(ValueError):
        validate_web_url(url)


def test_validate_web_url_accepts_http_and_https():
    assert validate_web_url("https://example.com/a?b=1") == "https://example.com/a?b=1"
    assert validate_web_url("http://example.com") == "http://example.com"


# -- 失败 -----------------------------------------------------------------


def test_missing_key_is_reported_as_a_clear_adapter_error():
    adapter = TavilyAdapter(api_key=None)

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_web_search("茅台", 5)

    assert "未配置" in str(exc.value)
    assert not exc.value.retryable


def test_adapter_uses_the_os_proxy_when_none_is_configured(monkeypatch):
    """httpx 只读取环境变量；requests（以及 akshare）还会读取操作系统的
    proxy。没有这一点，两者在配置了 proxy 的 host 上会走不同的路由。"""
    from finharness.data.adapters import tavily_adapter

    monkeypatch.setattr(
        tavily_adapter.urllib.request,
        "getproxies",
        lambda: {"https": "http://127.0.0.1:7890", "http": "http://127.0.0.1:7890"},
    )

    adapter = TavilyAdapter(api_key="k")

    assert adapter.proxy == "http://127.0.0.1:7890"


def test_explicit_proxy_overrides_the_os_proxy(monkeypatch):
    from finharness.data.adapters import tavily_adapter

    monkeypatch.setattr(
        tavily_adapter.urllib.request, "getproxies", lambda: {"https": "http://os:1"}
    )

    adapter = TavilyAdapter(api_key="k", proxy="http://explicit:2")

    assert adapter.proxy == "http://explicit:2"


def test_proxy_discovery_failure_is_tolerated(monkeypatch):
    """没有 proxy 支持的机器不得导致搜索失败。"""
    from finharness.data.adapters import tavily_adapter

    def boom():
        raise OSError("no proxy subsystem")

    monkeypatch.setattr(tavily_adapter.urllib.request, "getproxies", boom)

    assert TavilyAdapter(api_key="k").proxy is None


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
    """对于中文查询，topic=finance 是有问题的，因此会事先被拒绝。"""
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
