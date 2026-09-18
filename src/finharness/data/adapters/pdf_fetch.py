"""研报全文的本地 PDF 抓取（docs 03.4）。

研报的元数据来自 JSON API，但研报的 *正文* 只以 PDF 形式存在，而该 PDF 位于
一个会对非浏览器客户端返回 JavaScript 反爬校验的文档 CDN 上。本模块掌管代码
库中唯一一处与文档主机建立连接的地方，其编写原则是让这一攻击面尽可能小：

* **有界。** 重定向、墙钟时间和响应体大小都设有上限。
* **私有网段拦截。** 请求前先解析主机并对每个返回地址做网段检查，且在每次
  重定向跳转时重复该检查，因此一个文档 URL 无法把请求弹到内网或云元数据端点。
* **尽力而为的反爬。** 当收到校验页面时，会根据脚本自身的常量重新计算 cookie
  并在有界次数内重试。该方案会被不定期重新下发，且其脚本经过混淆，因此这里
  是刻意做成可恢复而非有保证的：遇到不熟悉的脚本时，把失败留给调用方，而不是
  抛出异常。

一个坦诚的局限：地址检查是在请求前即时解析域名，而不是固定对端，因此它并非
能抵御 DNS 重绑定（那样做要付出 TLS SNI 的代价）。
"""

from __future__ import annotations

import io
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple
from urllib.parse import urljoin, urlparse

from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.tavily_adapter import validate_web_url

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5
_DEFAULT_MAX_BYTES = 40 * 1024 * 1024
_DEFAULT_TIMEOUT_S = 30.0
# 即使在成功破解之后，校验也会被间歇性地重新下发，因此仅做一次 cookie 往返
# 是不够的；少量几次尝试足以清除瞬时情况，又不会把持续性的封禁变成重试风暴。
_DEFAULT_ATTEMPTS = 3

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class Fetched(NamedTuple):
    """一次已完成的跳转：响应头以及（受大小上限约束的）响应体。"""

    headers: dict[str, str]
    body: bytes


# 注入传输层，使得守卫逻辑、校验页处理和 PDF 解析都能在没有 socket 的情况下
# 进行测试。
FetchFn = Callable[..., "Fetched"]


def is_blocked_ip(raw: str) -> bool:
    """当地址不允许被本地请求访问时返回 True。

    覆盖回环、私有、链路本地（其中包含云元数据地址 169.254.169.254）、组播、
    保留以及未指定网段。任何无法解析的地址都被视为被拦截——一个我们无法归类的
    地址就不该去连接。
    """
    try:
        address = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        return True
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def assert_public_host(host: str) -> None:
    """解析 ``host``，只要任一地址为非公网就拒绝。"""
    if not host:
        raise AdapterError("网址缺少主机名，本地抓取已跳过")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise AdapterError(f"无法解析主机名：{host}") from exc
    for info in infos:
        if is_blocked_ip(str(info[4][0])):
            raise AdapterError(f"拒绝访问非公网地址：{host}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """把 3xx 转换为 HTTPError，以便每次跳转都能被手工重新校验。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _default_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect)


def fetch_bytes(
    url: str,
    *,
    cookie: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    opener: urllib.request.OpenerDirector | None = None,
) -> Fetched:
    """带守卫地 GET ``url``，并手动跟随重定向。

    ``cookie`` 仅在第一次跳转时设置，这对抗爬握手已经足够（服务器在同一 URL 上
    接受计算出的 cookie）。
    """
    client = opener or _default_opener()
    current = validate_web_url(url)
    headers_base = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    first = True
    for _ in range(_MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
            raise AdapterError(f"只支持 http/https 绝对网址：{current}")
        assert_public_host(parsed.hostname or "")

        headers = dict(headers_base)
        if first and cookie:
            headers["Cookie"] = cookie
        request = urllib.request.Request(current, headers=headers)
        try:
            response = client.open(request, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            if exc.code in _REDIRECT_CODES:
                location = exc.headers.get("Location")
                exc.close()
                if not location:
                    raise AdapterError(f"下载失败：重定向 {exc.code} 缺少目标地址") from exc
                current = urljoin(current, location)
                first = False
                continue
            raise AdapterError(
                f"下载失败：HTTP {exc.code}",
                retryable=exc.code == 429 or exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise AdapterError(f"下载失败：{exc}", retryable=True) from exc

        try:
            body = response.read(max_bytes + 1)
            response_headers = {k.lower(): v for k, v in response.headers.items()}
        finally:
            response.close()
        if len(body) > max_bytes:
            raise AdapterError("PDF 超过本地抓取大小上限，已放弃")
        return Fetched(headers=response_headers, body=body)

    raise AdapterError("下载失败：重定向次数过多")


# -- 反爬校验 ----------------------------------------------------------------
# 若干金融文档主机会对首次请求返回一段小脚本，用于设置 ``__tst_status`` 和
# ``EO_Bot_Ssid`` cookie 然后重新加载。这些 cookie 由嵌入在该脚本中的常量计算
# 得出。我们在此重新计算它们，而不是执行该脚本：这样做的好处仅限于使用该方案的
# 站点，且一旦解析失败，就把该 URL 交给提供方处理。

_CHALLENGE_MARKER = "EO_Bot_Ssid"
_ARRAY_RE = re.compile(r"_0x[0-9a-fA-F]+=\[(.*?)\];")
_ROTATE_RE = re.compile(r"\)\(_0x[0-9a-fA-F]+,(0x[0-9a-fA-F]+)\)")
_CONSTANTS_RE = re.compile(r"var e=\{(.*?)\},t=0;", re.DOTALL)
_NUMBER_RE = re.compile(r"([A-Za-z_$][\w$]*):(\d+)")
_OFFSET_RE = re.compile(r"\(t,(\d+)\)")


def solve_challenge(script: str) -> str | None:
    """计算校验所需的 cookie；若脚本结构陌生则返回 ``None``。

    设计上即为尽力而为：脚本经过混淆且会变化。遇到任何意外就返回 ``None``，
    可避免不熟悉的变体触发异常。
    """
    if _CHALLENGE_MARKER not in script:
        return None
    try:
        array_match = _ARRAY_RE.search(script)
        rotate_match = _ROTATE_RE.search(script)
        constants_match = _CONSTANTS_RE.search(script)
        offset_match = _OFFSET_RE.search(script)
        if not (array_match and rotate_match and constants_match and offset_match):
            return None

        names = json.loads("[" + array_match.group(1) + "]")
        if len(names) < 4:
            return None
        rotations = int(rotate_match.group(1), 16) % len(names)
        for _ in range(rotations):
            names.append(names.pop(0))

        constants = {
            key: int(value)
            for key, value in _NUMBER_RE.findall(constants_match.group(1))
        }
        # 轮换之后，访问器的前两个槽位和第四个槽位指向被求和的常量；第三个槽位
        # 指向（被忽略的）加法辅助函数。
        total = constants[names[0]] + constants[names[1]] + constants[names[3]]
        sid = offset_match.group(1)
    except (KeyError, ValueError, IndexError, TypeError):
        return None
    return f"__tst_status={total}#; EO_Bot_Ssid={sid}"


def _is_challenge(body: bytes) -> bool:
    """校验页是一个极小的脚本响应体，绝不会是 PDF。"""
    if body[:4] == b"%PDF":
        return False
    head = body[:2048].decode("utf-8", "replace")
    return _CHALLENGE_MARKER in head and "<script" in head


def fetch_pdf_bytes(
    url: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    attempts: int = _DEFAULT_ATTEMPTS,
    opener: urllib.request.OpenerDirector | None = None,
) -> bytes:
    """抓取 PDF，在文档主机返回反爬校验时予以破解。

    以有界次数重试，因为即使在成功破解之后，校验也会被间歇性地重新下发。
    """
    reason = "未知原因"
    for _ in range(max(1, attempts)):
        fetched = fetch_bytes(url, timeout_s=timeout_s, max_bytes=max_bytes, opener=opener)
        if fetched.body[:4] == b"%PDF":
            return fetched.body
        if not _is_challenge(fetched.body):
            content_type = fetched.headers.get("content-type") or "未知"
            raise AdapterError(f"该地址未返回 PDF（content-type：{content_type}）")

        cookie = solve_challenge(fetched.body.decode("utf-8", "replace"))
        if not cookie:
            reason = "无法解析反爬校验脚本"
            continue
        retried = fetch_bytes(
            url, cookie=cookie, timeout_s=timeout_s, max_bytes=max_bytes, opener=opener
        )
        if retried.body[:4] == b"%PDF":
            return retried.body
        reason = "反爬校验未通过"
    raise AdapterError(f"未能获取 PDF（{reason}）")


def extract_pdf_pages(data: bytes) -> list[str]:
    """逐页提取 PDF 的文本层，返回按页码顺序排列的列表。

    与 :func:`extract_pdf_text` 的区别在于**保留页边界**：分页读取（``read_pdf``）
    需要知道第 N 页从何开始，而合并后的整段文本无法回答这个问题。单页出错只丢该页，
    不让其余页随之丢失——扫描件里夹一页图片不该让整份文档变成"不可读"。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf 是已声明的依赖项
        raise AdapterError("缺少 pypdf，无法解析 PDF") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
    except Exception as exc:  # noqa: BLE001 - 文件格式错误属于明确的失败
        raise AdapterError(f"PDF 解析失败：{exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - 单页出错不能让其余内容丢失
            pages.append("")
    return pages


def extract_pdf_text(data: bytes) -> str:
    """逐页提取 PDF 的文本层，并合并为整段文本。

    没有文本层的 PDF（扫描件）会被如实报告为如此，而不是返回空内容："已下载但
    不可读" 与 "无此文档" 对阅读者而言是不同的答案。
    """
    text = "\n".join(extract_pdf_pages(data)).strip()
    if not text:
        raise AdapterError("PDF 已下载但未抽取到文本（可能是扫描件，需 OCR）")
    return text


def save_pdf_bytes(data: bytes, directory: str | Path) -> Path:
    """把 PDF 落盘到内容寻址的文件名，并返回其路径。

    文件名取自内容的 sha256 前缀：同一份文档重复抓取只写一次（天然去重），且内容
    不可变、无需失效逻辑。先写临时文件再 ``replace``，使读取方永远看不到半个文件。
    落盘是让"全文"在上下文之外仍然可用的前提——上下文只承载有界预览与这个句柄。
    """
    import hashlib
    import os
    import tempfile

    digest = hashlib.sha256(data).hexdigest()[:16]
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest}.pdf"
    if target.is_file() and target.stat().st_size == len(data):
        return target
    handle, temp_name = tempfile.mkstemp(dir=target_dir, suffix=".part")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp_name, target)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


_PAGE_RANGE_RE = re.compile(r"^\s*(\d*)\s*-\s*(\d*)\s*$")


def parse_page_range(spec: str | None, total: int) -> tuple[int, int]:
    """把 ``"1-3"`` / ``"5"`` / ``"2-"`` / ``None`` 解析为闭区间 ``(first, last)``。

    页码从 1 开始，且会被夹到 ``total`` 之内。非法或越界的取值由调用方转述，
    这里只负责在给定时给出一个可用的区间。
    """
    if spec is None or not str(spec).strip():
        return 1, min(total, 1) if total else 1
    text = str(spec).strip()
    match = _PAGE_RANGE_RE.match(text)
    if match:
        start = int(match.group(1)) if match.group(1) else 1
        end = int(match.group(2)) if match.group(2) else total
        return start, end
    if text.isdigit():
        page = int(text)
        return page, page
    raise AdapterError(f"无法识别的页码范围：{spec}（可用 '3'、'1-3'、'2-'）")


def read_pdf_pages(path: str | Path, pages: str | None = None) -> tuple[list[str], int, int, int]:
    """读取本地 PDF 的指定页，返回 ``(页文本列表, 首页, 末页, 总页数)``。

    越界会被如实修剪并返回实际的页区间，使调用方能说明"你要的第 9 页只有 6 页"，
    而不是静默返回一个空列表。
    """
    target = Path(path)
    if not target.is_file():
        raise AdapterError(f"文件不存在：{path}")
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise AdapterError(f"读取 PDF 失败：{exc}") from exc

    all_pages = extract_pdf_pages(data)
    total = len(all_pages)
    if total == 0:
        raise AdapterError("PDF 不含任何页面")
    first, last = parse_page_range(pages, total)
    if first < 1 or first > total:
        raise AdapterError(f"页码超出范围：第 {first} 页（共 {total} 页）")
    last = max(first, min(last, total))
    return all_pages[first - 1 : last], first, last, total


def fetch_pdf_text(url: str, **kwargs) -> str:
    """便捷函数：抓取一份研报 PDF 并返回其文本层。"""
    return extract_pdf_text(fetch_pdf_bytes(url, **kwargs))
