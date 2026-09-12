"""Resolve the active provider: database config first, settings preset as fallback."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from finharness.config.settings import ProviderSettings, Settings
from finharness.config.store import ConfigStore, ProviderConfigRecord
from finharness.provider.base import Provider
from finharness.provider.registry import build_provider_from_fields


class NotConfigured(RuntimeError):
    """Raised when neither the database nor settings yields a usable provider."""


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    configured: bool
    source: str | None
    name: str | None
    kind: str | None
    model: str | None
    base_url: str | None
    has_key: bool


def _default_timeouts(settings: Settings, kind: str) -> tuple[float, float]:
    """Borrow timeout defaults from any preset sharing the requested kind."""
    for preset in settings.providers.values():
        if preset.kind == kind:
            return preset.first_byte_timeout_s, preset.idle_timeout_s
    return 30.0, 60.0


def _settings_has_key(preset: ProviderSettings) -> bool:
    return bool(preset.env_key and os.getenv(preset.env_key))


class ProviderResolver:
    """Caches the active provider and rebuilds it only when configuration changes."""

    def __init__(
        self,
        *,
        store_factory: Callable[[], ConfigStore],
        settings: Settings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._store_factory = store_factory
        self._settings = settings
        self._client = client
        self._cached_key: tuple | None = None
        self._cached_provider: Provider | None = None

    @property
    def _store(self) -> ConfigStore:
        return self._store_factory()

    def invalidate(self) -> None:
        self._cached_key = None
        self._cached_provider = None

    def status(self) -> ProviderStatus:
        active = self._store.get_active()
        if active is not None:
            return ProviderStatus(
                configured=active.kind == "fake" or self._store.resolve_key(active.id) is not None,
                source="database",
                name=active.name,
                kind=active.kind,
                model=active.model,
                base_url=active.base_url,
                has_key=active.has_key or bool(active.env_key),
            )
        preset = self._settings.providers.get(self._settings.model.provider)
        if preset is None:
            return ProviderStatus(
                configured=False, source=None, name=None, kind=None,
                model=None, base_url=None, has_key=False,
            )
        available = preset.kind == "fake" or _settings_has_key(preset)
        return ProviderStatus(
            configured=available,
            source="settings",
            name=self._settings.model.provider,
            kind=preset.kind,
            model=self._settings.model.model_name,
            base_url=preset.base_url,
            has_key=available,
        )

    def current(self) -> Provider:
        active = self._store.get_active()
        if active is not None:
            return self._from_database(active)
        return self._from_settings()

    def _from_database(self, active: ProviderConfigRecord) -> Provider:
        cache_key = ("database", active.id, active.updated_at)
        if self._cached_key == cache_key and self._cached_provider is not None:
            return self._cached_provider
        api_key = self._store.resolve_key(active.id)
        if active.kind != "fake" and not api_key:
            raise NotConfigured(
                f"配置「{active.name}」缺少可用 API Key，请重新填写或设置环境变量"
            )
        first_byte, idle = _default_timeouts(self._settings, active.kind)
        provider = build_provider_from_fields(
            kind=active.kind,
            base_url=active.base_url,
            api_key=api_key,
            model=active.model,
            temperature=self._settings.model.temperature,
            max_tokens=self._settings.model.max_tokens,
            first_byte_timeout_s=first_byte,
            idle_timeout_s=idle,
            client=self._client,
        )
        self._cached_key = cache_key
        self._cached_provider = provider
        return provider

    def _from_settings(self) -> Provider:
        preset = self._settings.providers.get(self._settings.model.provider)
        if preset is None:
            raise NotConfigured("未配置供应商，请先在设置中配置模型供应商")
        cache_key = ("settings", self._settings.model.provider, self._settings.model.model_name)
        if self._cached_key == cache_key and self._cached_provider is not None:
            return self._cached_provider
        if preset.kind == "fake":
            provider = build_provider_from_fields(
                kind="fake", base_url=None, api_key=None,
                model=self._settings.model.model_name, client=self._client,
            )
        else:
            if not _settings_has_key(preset):
                raise NotConfigured(
                    "未配置供应商，请先在设置中配置模型供应商，"
                    f"或设置环境变量 {preset.env_key}"
                )
            provider = build_provider_from_fields(
                kind=preset.kind,
                base_url=preset.base_url,
                api_key=os.environ[preset.env_key],
                model=self._settings.model.model_name,
                temperature=self._settings.model.temperature,
                max_tokens=self._settings.model.max_tokens,
                api_version=preset.api_version,
                first_byte_timeout_s=preset.first_byte_timeout_s,
                idle_timeout_s=preset.idle_timeout_s,
                client=self._client,
            )
        self._cached_key = cache_key
        self._cached_provider = provider
        return provider
