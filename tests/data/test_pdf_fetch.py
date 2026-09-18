"""本地 PDF 抓取：防护、反爬 challenge、文本抽取。

该模块是数据层中唯一会向文档 host 建立连接的地方，因此这些测试主要关注
它*拒绝*做什么。transport 是注入的，DNS 被打桩，所以这里不会触及网络。
"""

from __future__ import annotations

import socket

import pytest

from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.pdf_fetch import (
    assert_public_host,
    extract_pdf_text,
    fetch_bytes,
    fetch_pdf_bytes,
    is_blocked_ip,
    solve_challenge,
)
from tests.data.pdf_fixtures import make_pdf


def allow_all_dns(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
    )


def response(body: bytes, headers: dict | None = None):
    class _Response:
        def __init__(self):
            self.headers = headers or {"Content-Type": "application/pdf"}

        def read(self, n=-1):
            return body

        def close(self):
            pass

    return _Response()


# challenge 脚本抓取自 pdf.dfcfw.com，形态与原样一致。其中的
# 常量每次响应都是随机的，因此期望的 cookie 由这份副本计算得出。
CHALLENGE = (
    '<script>function a(a){function n(){for(var a={wQzOV:_0x649a("0x4"),'
    'iTyzs:function(a,n){return a+n}},n=a[_0x649a("0x5")][_0x649a("0x6")]("|"),e=0;;)'
    '{switch(n[e++]){case"0":t+="EO_Bot_Ssid=";continue;case"1":return t;'
    'case"2":t+="";continue;case"3":t=a[_0x649a("0x7")](t,3744923648);continue;'
    'case"4":var t="";continue}break}}var e={WTKkN:2299133025,bOYDu:14118781,'
    'dtzqS:function(a,n){return a+n},wyeCN:1772492478,pCQRM:function(a){return a()}},'
    't=0;return t+=e[_0x649a("0x0")],t+=e[_0x649a("0x1")],'
    't=e[_0x649a("0x2")](t,e[_0x649a("0x3")]),[t,e[_0x649a("0x8")](n)][a]}'
    'var _0x49a6=["wyeCN","4|2|0|3|1","wQzOV","split","iTyzs","pCQRM","cookie",'
    '"location.href=location.href.replace(/[?|&]tads/, \'\')","WTKkN","bOYDu","dtzqS"];'
    "(function(a,n){var e=function(n){for(;--n;)a.push(a.shift())};e(++n)})(_0x49a6,0x147);"
    'var _0x649a=function(a,n){a-=0;var e=_0x49a6[a];return e};'
    'document[_0x649a("0x9")]="__tst_status="+a(0)+"#;"</script>'
)


# -- SSRF 防护 --------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",            # 回环地址
        "10.0.0.5",             # 私有地址
        "192.168.1.1",          # 私有地址
        "172.16.0.1",           # 私有地址
        "169.254.169.254",      # 云元数据地址
        "0.0.0.0",              # 未指定地址
        "::1",                  # IPv6 回环地址
        "fc00::1",              # IPv6 唯一本地地址
        "not-an-ip",            # 无法解析 -> 视为已阻止
    ],
)
def test_non_public_addresses_are_blocked(address):
    assert is_blocked_ip(address) is True


@pytest.mark.parametrize("address", ["93.184.216.34", "8.8.8.8", "1.1.1.1"])
def test_public_addresses_are_allowed(address):
    assert is_blocked_ip(address) is False


def test_a_literal_private_host_is_refused(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 0, 0, "", ("127.0.0.1", 0))]
    )
    with pytest.raises(AdapterError) as exc:
        assert_public_host("localhost")

    assert "非公网" in str(exc.value)


def test_an_unresolvable_host_is_an_adapter_error(monkeypatch):
    def boom(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(AdapterError):
        assert_public_host("nope.invalid")


# -- 传输 ----------------------------------------------------------------


def test_fetch_bytes_reads_a_public_response(monkeypatch):
    allow_all_dns(monkeypatch)
    body = make_pdf()

    class FakeOpener:
        def open(self, request, timeout=None):
            return response(body)

    fetched = fetch_bytes("https://host/a.pdf", opener=FakeOpener())

    assert fetched.body == body
    assert fetched.headers["content-type"] == "application/pdf"


def test_fetch_bytes_revalidates_each_redirect_hop(monkeypatch):
    """公网 URL 不得能把请求重定向进内网。"""
    from urllib.error import HTTPError

    calls = {"n": 0}

    def fake_getaddrinfo(host, *a, **k):
        calls["n"] += 1
        # 第一跳为公网，第二跳（重定向后）为私有地址。
        address = "93.184.216.34" if calls["n"] == 1 else "127.0.0.1"
        return [(socket.AF_INET, 0, 0, "", (address, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    class RedirectingOpener:
        def open(self, request, timeout=None):
            raise HTTPError(
                request.full_url, 302, "Found",
                {"Location": "http://internal.local/secret.pdf"}, None,
            )

    with pytest.raises(AdapterError) as exc:
        fetch_bytes("https://public.example/a.pdf", opener=RedirectingOpener())

    assert "非公网" in str(exc.value)


def test_fetch_bytes_follows_a_public_redirect(monkeypatch):
    allow_all_dns(monkeypatch)
    from urllib.error import HTTPError

    body = make_pdf()
    seen = []

    class FakeOpener:
        def open(self, request, timeout=None):
            seen.append(request.full_url)
            if len(seen) == 1:
                raise HTTPError(
                    request.full_url, 302, "Found",
                    {"Location": "https://cdn.example/real.pdf"}, None,
                )
            return response(body)

    fetched = fetch_bytes("https://public.example/a.pdf", opener=FakeOpener())

    assert fetched.body == body
    assert seen == ["https://public.example/a.pdf", "https://cdn.example/real.pdf"]


def test_oversized_response_is_refused(monkeypatch):
    allow_all_dns(monkeypatch)

    class FakeOpener:
        def open(self, request, timeout=None):
            return response(b"x" * 1000)

    with pytest.raises(AdapterError) as exc:
        fetch_bytes("https://host/a.pdf", max_bytes=10, opener=FakeOpener())

    assert "大小上限" in str(exc.value)


# -- 反爬 challenge ----------------------------------------------------------------


def test_challenge_cookies_are_computed_from_the_script():
    # 数组旋转之后，访问器读取的是 WTKkN、bOYDu 和 wyeCN。
    assert 2299133025 + 14118781 + 1772492478 == 4085744284
    assert solve_challenge(CHALLENGE) == (
        "__tst_status=4085744284#; EO_Bot_Ssid=3744923648"
    )


def test_an_unfamiliar_challenge_returns_none_instead_of_raising():
    """该脚本经过混淆且会变化；未知的形态绝不能导致崩溃。"""
    assert solve_challenge("<script>totally different</script>") is None
    assert solve_challenge("<script>EO_Bot_Ssid but no array</script>") is None
    assert solve_challenge("") is None


# -- 抽取 ---------------------------------------------------------------


def test_extract_pdf_text_returns_the_text_layer():
    assert "Hello Direct PDF" in extract_pdf_text(make_pdf())


def test_a_malformed_pdf_is_an_adapter_error():
    with pytest.raises(AdapterError) as exc:
        extract_pdf_text(b"%PDF-1.4\nnot really a pdf")

    assert "PDF" in str(exc.value)


# -- fetch_pdf_bytes：challenge 会重新下发，因此重试次数有上限 ---------------


def test_fetch_pdf_bytes_retries_the_challenge_then_succeeds(monkeypatch):
    allow_all_dns(monkeypatch)
    body = make_pdf()
    seen = []

    class FakeOpener:
        def open(self, request, timeout=None):
            seen.append(request.headers.get("Cookie"))
            # 第一个请求会收到 challenge；随后携带 cookie 重试即可成功。
            if len(seen) == 1:
                return response(CHALLENGE.encode())
            return response(body)

    result = fetch_pdf_bytes("https://host/a.pdf", opener=FakeOpener(), attempts=3)

    assert result == body
    assert seen[0] is None  # 第一跳不携带 cookie
    assert seen[-1].startswith("__tst_status=")


def test_fetch_pdf_bytes_gives_up_after_the_attempt_budget(monkeypatch):
    allow_all_dns(monkeypatch)
    calls = {"n": 0}

    class FakeOpener:
        def open(self, request, timeout=None):
            calls["n"] += 1
            return response(CHALLENGE.encode())

    with pytest.raises(AdapterError) as exc:
        fetch_pdf_bytes("https://host/a.pdf", opener=FakeOpener(), attempts=3)

    assert "未能获取 PDF" in str(exc.value)
    # 每次尝试包含一次 fetch 加一次带 cookie 的重试。
    assert calls["n"] == 6


def test_a_non_pdf_body_is_reported_without_solving(monkeypatch):
    allow_all_dns(monkeypatch)

    class FakeOpener:
        def open(self, request, timeout=None):
            return response(b"<html>login required</html>", {"content-type": "text/html"})

    with pytest.raises(AdapterError) as exc:
        fetch_pdf_bytes("https://host/a.pdf", opener=FakeOpener())

    assert "未返回 PDF" in str(exc.value)


# -- 分页与落盘：read_pdf / summarize_document 的底层能力 ----------------------


def test_extract_pdf_pages_keeps_page_boundaries():
    """分页读取需要页边界，合并后的整段文本无法回答"第 N 页从哪开始"。"""
    from finharness.data.adapters.pdf_fetch import extract_pdf_pages
    from tests.data.pdf_fixtures import make_multi_page_pdf

    pages = extract_pdf_pages(make_multi_page_pdf(["zqxalpha", "zqxbeta", "zqxgamma"]))

    assert len(pages) == 3
    assert "zqxalpha" in pages[0]
    assert "zqxbeta" in pages[1]
    assert "zqxgamma" in pages[2]


def test_extract_pdf_text_still_joins_all_pages():
    from finharness.data.adapters.pdf_fetch import extract_pdf_text
    from tests.data.pdf_fixtures import make_multi_page_pdf

    text = extract_pdf_text(make_multi_page_pdf(["zqxone", "zqxtwo"]))

    assert "zqxone" in text and "zqxtwo" in text


def test_a_blank_pdf_is_reported_as_unreadable_not_empty():
    """扫描件是"已下载但不可读"，与"没有内容"是不同的事实。"""
    from finharness.data.adapters.pdf_fetch import extract_pdf_text

    with pytest.raises(AdapterError) as exc:
        extract_pdf_text(make_pdf(""))

    assert "未抽取到文本" in str(exc.value)


def test_save_pdf_bytes_is_content_addressed_and_idempotent(tmp_path):
    """同一份内容写两次只落一个文件：抓取重复不导致磁盘膨胀。"""
    from finharness.data.adapters.pdf_fetch import save_pdf_bytes

    data = make_pdf("zqxsame")
    first = save_pdf_bytes(data, tmp_path)
    second = save_pdf_bytes(data, tmp_path)

    assert first == second
    assert first.is_file()
    assert first.read_bytes() == data
    assert len(list(tmp_path.glob("*.pdf"))) == 1
    # 不同的内容得到不同的文件名。
    other = save_pdf_bytes(make_pdf("zqxother"), tmp_path)
    assert other != first
    assert len(list(tmp_path.glob("*.pdf"))) == 2


def test_save_pdf_bytes_leaves_no_partial_file_behind(tmp_path):
    from finharness.data.adapters.pdf_fetch import save_pdf_bytes

    save_pdf_bytes(make_pdf("zqxpartial"), tmp_path)

    assert not list(tmp_path.glob("*.part"))


def test_parse_page_range_handles_the_supported_forms():
    from finharness.data.adapters.pdf_fetch import parse_page_range

    assert parse_page_range(None, 10) == (1, 1)
    assert parse_page_range("3", 10) == (3, 3)
    assert parse_page_range("2-5", 10) == (2, 5)
    assert parse_page_range("4-", 10) == (4, 10)
    assert parse_page_range("-3", 10) == (1, 3)


def test_parse_page_range_rejects_nonsense():
    from finharness.data.adapters.pdf_fetch import parse_page_range

    with pytest.raises(AdapterError):
        parse_page_range("abc", 10)


def test_read_pdf_pages_clamps_an_overlong_end(tmp_path):
    """末页超出总页数时修剪到实际的最后一页，而不是返回空。"""
    from finharness.data.adapters.pdf_fetch import read_pdf_pages
    from tests.data.pdf_fixtures import make_multi_page_pdf

    target = tmp_path / "r.pdf"
    target.write_bytes(make_multi_page_pdf(["zqxa", "zqxb"]))

    pages, first, last, total = read_pdf_pages(target, "2-99")

    assert (first, last, total) == (2, 2, 2)
    assert "zqxb" in pages[0]


def test_read_pdf_pages_refuses_a_page_past_the_end(tmp_path):
    from finharness.data.adapters.pdf_fetch import read_pdf_pages
    from tests.data.pdf_fixtures import make_multi_page_pdf

    target = tmp_path / "r.pdf"
    target.write_bytes(make_multi_page_pdf(["zqxa"]))

    with pytest.raises(AdapterError) as exc:
        read_pdf_pages(target, "9")

    assert "超出范围" in str(exc.value)
