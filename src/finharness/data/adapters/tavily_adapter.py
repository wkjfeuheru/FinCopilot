"""Tavily 适配器：外部网络搜索（docs 03.4）。

出站搜索请求由 Tavily 的服务器发出，因此本进程从不与用户提供的地址建立连接，
这条路径上不存在 SSRF 攻击面。代价则是能力受限——内网和本地地址不可达，且每次
查询都会披露给 Tavily——对于公开的金融研究这两点都可以接受，而这也是本功能
唯一面向的用途。

结果以一个小型 DataFrame 返回，从而复用项目现有的缓存、引用和适配器回退机制，
而不是另起一条并行路径。

``validate_web_url`` 也位于本模块，并被本地 PDF 抓取复用，使协议筛选只有一处
定义而不是两处。
"""

from __future__ import annotations

import urllib.request
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd

from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult

SEARCH_INTERFACE = "search"

# 搜索表格所携带的列。刻意保持精简：title 和 url 是阅读者引用的内容，content
# 是模型用来判断相关性的摘要片段。
SEARCH_COLUMNS = ("title", "url", "content")

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Tavily 的 ``topic="finance"`` 对中文查询会返回严重不相关的结果——已对照线上
# API 验证：一个茅台营收查询返回了综合格斗和艺术品拍卖页面。由于本项目使用中文
# 搜索，只接受真正表现正常的值；``general`` 是默认值，``news`` 对时效性条目
# 没有问题。
_ALLOWED_TOPICS = frozenset({"general", "news"})


def validate_web_url(url: str) -> str:
    """拒绝任何非绝对 http(s) URL。

    这并非 SSRF 防御——发出请求的是 Tavily 而不是我们。这是输入卫生：阻止
    ``file://``、``ftp://`` 以及相对路径被当作可抓取的页面发送给提供方。
    """
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise ValueError(f"只支持 http/https 绝对网址：{url}")
    return url.strip()


def system_proxy() -> str | None:
    """操作系统所配置的代理；仅靠 ``httpx`` 是看不到它的。

    ``requests``（A 股适配器所用）会调用 ``getproxies()``，因此能读取 Windows
    注册表中的代理；而 ``httpx`` 在 ``trust_env=True`` 时只读取环境变量。在某台
    把系统代理作为外网出口的主机上，这一差异会导致同一个进程能访问国内数据源，
    却与搜索提供方的每个连接都失败。在此读取它可让两者保持一致。

    与其他 ``httpx`` 适配器共享，使本包中每个出站客户端都走同一条路由。
    """
    try:
        proxies = urllib.request.getproxies()
    except Exception:  # noqa: BLE001 - 代理探测绝不能使调用失败
        return None
    return proxies.get("https") or proxies.get("http") or None


def _message_from(response: httpx.Response) -> str:
    """Tavily 把错误文本放在 ``detail.error`` 下。"""
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
    """通过 Tavily API 进行搜索与提取；一个密钥同时服务两者。"""

    name = "tavily"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.tavily.com",
        timeout_s: float = 30.0,
        proxy: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # 显式指定的代理优先；否则回退到系统代理，使本适配器与 A 股数据源走
        # 相同的路由。
        self.proxy = proxy or system_proxy()
        # 可注入，以便测试在不触碰网络的情况下验证解析逻辑。
        self._client = client

    # -- 传输层 ---------------------------------------------------------------
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """向 Tavily 端点 POST 请求体，并把失败统一转换为 AdapterError。"""
        if not self.api_key:
            raise AdapterError(
                "未配置联网检索密钥（TAVILY_API_KEY）：web_search 当前不可用"
            )
        headers = {"Authorization": "Bearer " + self.api_key}
        body = {**payload, "api_key": self.api_key}
        try:
            if self._client is not None:
                response = self._client.post(
                    self.base_url + path, json=body, headers=headers, timeout=self.timeout_s
                )
            else:
                with httpx.Client(timeout=self.timeout_s, proxy=self.proxy) as client:
                    response = client.post(
                        self.base_url + path, json=body, headers=headers
                    )
        except httpx.HTTPError as exc:
            raise AdapterError(f"联网检索请求失败：{exc}", retryable=True) from exc

        if response.status_code >= 400:
            # 429 和 5xx 值得再试一次；认证和配额错误则不值得。
            retryable = response.status_code == 429 or response.status_code >= 500
            raise AdapterError(
                f"联网检索返回 {response.status_code}：{_message_from(response)}",
                retryable=retryable,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise AdapterError("联网检索返回了非 JSON 响应") from exc

    # -- 语义化抓取 -----------------------------------------------------------
    def fetch_web_search(
        self,
        query: str,
        top_n: int,
        topic: str | None = None,
        time_range: str | None = None,
    ) -> FetchResult:
        """执行一次联网检索，返回 title/url/content 三列的结果表。"""
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(int(top_n), 20)),
            "search_depth": "basic",
            # 仅取摘要片段：整页内容会让搜索引擎支配上下文预算。搜索返回的是
            # 片段；完整文档是另一件事（见研报工具）。
            "include_raw_content": False,
            "include_answer": False,
        }
        if topic:
            if topic not in _ALLOWED_TOPICS:
                # 拒绝才是诚实的失败：把它发出去会返回看似合理但与主题无关的
                # 页面，而模型可能引用这些页面。
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
