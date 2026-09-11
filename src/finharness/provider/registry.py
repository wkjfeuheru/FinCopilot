"""Provider construction from validated settings."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from finharness.config.settings import Settings, SettingsError
from finharness.provider.fake import FakeProvider
from finharness.provider.openai_compat import OpenAICompatProvider
from finharness.provider.anthropic_compat import AnthropicCompatProvider


def build_provider(
    path: str | Path = "settings.json",
    *,
    client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
):
    settings = settings or Settings.from_file(path)
    settings.validate(require_api_key=True)
    config = settings.providers[settings.model.provider]
    if config.kind == "fake":
        return FakeProvider(["FakeProvider is enabled explicitly."])
    if config.kind == "anthropic_compat":
        return AnthropicCompatProvider(
            base_url=config.base_url,
            api_key=os.environ[config.env_key],
            model=settings.model.model_name,
            api_version=config.api_version or "2023-06-01",
            temperature=settings.model.temperature,
            max_tokens=settings.model.max_tokens,
            first_byte_timeout_s=config.first_byte_timeout_s,
            idle_timeout_s=config.idle_timeout_s,
            client=client or httpx.AsyncClient(timeout=httpx.Timeout(config.idle_timeout_s, connect=config.first_byte_timeout_s)),
        )
    if config.kind != "openai_compat":
        raise SettingsError(f"Unsupported provider kind: {config.kind}")
    return OpenAICompatProvider(
        base_url=config.base_url,
        api_key=os.environ[config.env_key],
        model=settings.model.model_name,
        temperature=settings.model.temperature,
        max_tokens=settings.model.max_tokens,
        first_byte_timeout_s=config.first_byte_timeout_s,
        idle_timeout_s=config.idle_timeout_s,
        client=client or httpx.AsyncClient(timeout=httpx.Timeout(config.idle_timeout_s, connect=config.first_byte_timeout_s)),
    )
