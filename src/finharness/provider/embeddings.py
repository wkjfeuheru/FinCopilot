"""远程 /embeddings 客户端（docs 03.6.4 LTM 语义记忆）。

provider 层只有流式 chat completions，没有嵌入能力，因此这里单写一个**非流式**
的 OpenAI 兼容客户端：POST ``{base_url}/embeddings``，取 ``data[*].embedding``。

它是可选的：未配置 ``ltm.embeddings.base_url`` 时 ``build_embedder`` 返回
``None``，语义记忆照常写入（检索退化为键匹配）。任何传输/解析失败都只返回
``None`` 而不是抛出——记忆的语义召回是增强项，不是主链路。
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from finharness.config.settings import Settings


class Embedder:
    """OpenAI 兼容 /embeddings 端点的最小封装。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_s: float = 30.0,
        proxy: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        # 与搜索适配器同一路由口径：显式代理优先，否则跟随系统代理。
        if proxy is None:
            try:
                from finharness.data.adapters.tavily_adapter import system_proxy

                proxy = system_proxy()
            except Exception:  # noqa: BLE001 - 代理探测失败不是错误
                proxy = None
        self.proxy = proxy
        # 可注入，便于测试在不触碰网络时验证解析。
        self._client = client
        # 由首次成功响应推断的维度，用于建向量集合并校验后续一致性。
        self.dim = 0

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """把一批文本编码为向量；失败返回 ``None``（调用方据此降级）。

        单批失败不影响调用方——语义记忆的向量只是检索增强，写入本身不该
        因为嵌入服务抖动而失败。
        """
        clean = [text for text in texts if text and text.strip()]
        if not clean:
            return []
        payload: dict[str, Any] = {"model": self.model, "input": clean}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        try:
            if self._client is not None:
                response = self._client.post(
                    self.base_url + "/embeddings", json=payload, headers=headers,
                    timeout=self.timeout_s,
                )
            else:
                with httpx.Client(timeout=self.timeout_s, proxy=self.proxy) as client:
                    response = client.post(
                        self.base_url + "/embeddings", json=payload, headers=headers
                    )
            if response.status_code >= 400:
                return None
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        vectors = _parse_embeddings(body)
        if not vectors:
            return None
        if self.dim == 0:
            self.dim = len(vectors[0])
        # 维度必须一致，否则向量库里的距离计算毫无意义。
        if any(len(vector) != self.dim for vector in vectors):
            return None
        return vectors

    def embed_one(self, text: str) -> list[float] | None:
        vectors = self.embed([text])
        if not vectors:
            return None
        return vectors[0]


def _parse_embeddings(body: Any) -> list[list[float]]:
    """解析 OpenAI 兼容响应；按 ``index`` 排序以恢复输入顺序。

    某些端点会并发返回、乱序给出条目，而调用方按输入顺序配对，因此必须
    显式排序而不是依赖响应顺序。
    """
    if not isinstance(body, dict):
        return []
    items = body.get("data")
    if not isinstance(items, list):
        return []
    ordered: list[tuple[int, list[float]]] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        vector = item.get("embedding")
        if not isinstance(vector, list):
            continue
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError):
            continue
        index = item.get("index")
        ordered.append((int(index) if isinstance(index, int) else position, values))
    ordered.sort(key=lambda pair: pair[0])
    return [values for _index, values in ordered]


def build_embedder(settings: Settings) -> Embedder | None:
    """按配置构建嵌入器；未配置端点时返回 ``None``（语义召回关闭）。"""
    config = settings.ltm.embeddings
    if not config.base_url:
        return None
    api_key = os.getenv(config.env_key) if config.env_key else None
    return Embedder(
        base_url=config.base_url,
        api_key=api_key,
        model=config.model_name,
        timeout_s=float(config.timeout_s),
    )


__all__ = ["Embedder", "build_embedder"]
