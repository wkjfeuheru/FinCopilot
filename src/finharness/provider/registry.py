"""根据校验后的配置构建 provider。"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from finharness.config.settings import Settings, SettingsError
from finharness.provider.anthropic_compat import AnthropicCompatProvider
from finharness.provider.fake import FakeProvider
from finharness.provider.openai_compat import OpenAICompatProvider


def _build_client(*, first_byte_timeout_s: float, idle_timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(idle_timeout_s, connect=first_byte_timeout_s)
    )


def build_provider_from_fields(
    *,
    kind: str,
    base_url: str | None,
    api_key: str | None,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    api_version: str | None = None,
    first_byte_timeout_s: float = 30.0,
    idle_timeout_s: float = 60.0,
    client: httpx.AsyncClient | None = None,
):
    """根据显式字段组装 provider，绕过预设查找。"""
    if kind == "fake":
        return FakeProvider(["FakeProvider is enabled explicitly."])
    if kind not in {"openai_compat", "anthropic_compat"}:
        raise SettingsError(f"Unsupported provider kind: {kind}")
    if not api_key:
        raise SettingsError("缺少 API Key")
    shared_client = client or _build_client(
        first_byte_timeout_s=first_byte_timeout_s, idle_timeout_s=idle_timeout_s
    )
    if kind == "anthropic_compat":
        return AnthropicCompatProvider(
            base_url=base_url or "",
            api_key=api_key,
            model=model,
            api_version=api_version or "2023-06-01",
            temperature=temperature,
            max_tokens=max_tokens,
            first_byte_timeout_s=first_byte_timeout_s,
            idle_timeout_s=idle_timeout_s,
            client=shared_client,
        )
    return OpenAICompatProvider(
        base_url=base_url or "",
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        first_byte_timeout_s=first_byte_timeout_s,
        idle_timeout_s=idle_timeout_s,
        client=shared_client,
    )


def build_provider(
    path: str | Path = "settings.json",
    *,
    client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
):
    """从 settings 文件（或给定 Settings）构建当前激活的 provider。"""
    settings = settings or Settings.from_file(path)
    settings.validate(require_api_key=True)
    config = settings.providers[settings.model.provider]
    if config.kind == "fake":
        return FakeProvider(["FakeProvider is enabled explicitly."])
    return build_provider_from_fields(
        kind=config.kind,
        base_url=config.base_url,
        api_key=os.environ[config.env_key],
        model=settings.model.model_name,
        temperature=settings.model.temperature,
        max_tokens=settings.model.max_tokens,
        api_version=config.api_version,
        first_byte_timeout_s=config.first_byte_timeout_s,
        idle_timeout_s=config.idle_timeout_s,
        client=client,
    )
