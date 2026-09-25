"""用于 provider 配置管理的 HTTP 路由。

密钥只写入加密存储，绝不回显：
响应暴露一个 ``has_key`` 布尔值而不是密钥值。
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from finharness.auth.store import CurrentUser
from finharness.config.settings import Settings
from finharness.config.store import (
    ActiveConfigDeleteError,
    ConfigNotFound,
    ConfigStore,
    DuplicateConfigName,
)
from finharness.provider.base import Provider
from finharness.provider.errors import ProviderError
from finharness.provider.policy import validate_user_provider_url
from finharness.provider.registry import build_provider_from_fields
from finharness.provider.resolver import ProviderResolver
from finharness.types import ModelUsage, Msg, StreamEvent

SUPPORTED_KINDS = {"openai_compat", "anthropic_compat", "fake"}
PROBE_TIMEOUT_S = 20.0
_MISSING_SECRET_MESSAGE = "请填写 API Key 或提供可用的环境变量"


class ConfigPayload(BaseModel):
    name: str
    kind: str
    base_url: str | None = None
    model: str
    env_key: str | None = None
    api_key: str | None = None
    activate: bool = True


class ProbePayload(BaseModel):
    kind: str
    base_url: str | None = None
    model: str
    env_key: str | None = None
    api_key: str | None = None
    config_id: int | None = None


def _field_errors(
    payload: ConfigPayload | ProbePayload,
    settings: Settings,
) -> list[dict[str, str]]:
    """校验配置/探测载荷的字段，返回字段级错误列表。"""
    errors: list[dict[str, str]] = []
    if isinstance(payload, ConfigPayload) and not payload.name.strip():
        errors.append({"field": "name", "message": "配置名称不能为空"})
    if not payload.model.strip():
        errors.append({"field": "model", "message": "模型名称不能为空"})
    if payload.kind not in SUPPORTED_KINDS:
        errors.append({"field": "kind", "message": f"不支持的协议类型：{payload.kind}"})
    url_error = validate_user_provider_url(payload.base_url, payload.kind, settings)
    if url_error is not None:
        errors.append({"field": "base_url", "message": url_error})
    return errors


def _missing_secret_error() -> dict[str, str]:
    return {"field": "api_key", "message": _MISSING_SECRET_MESSAGE}


def _resolve_secret(explicit: str | None, env_name: str | None) -> str | None:
    """优先使用显式提供的密钥，其次是指定的环境变量。"""
    if explicit:
        return explicit
    if env_name:
        return os.getenv(env_name)
    return None


def _effective_secret(store: ConfigStore, payload, *, config_id: int | None, user_id: str = "") -> str | None:
    """为未保存的载荷解析密钥，编辑时复用已存储的值。"""
    secret = _resolve_secret(payload.api_key, payload.env_key)
    if secret:
        return secret
    if config_id is not None and store.get(config_id, user_id=user_id) is not None:
        return store.resolve_key(config_id, user_id=user_id)
    return None


def _record_payload(record, *, has_key: bool) -> dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "kind": record.kind,
        "base_url": record.base_url,
        "model": record.model,
        "env_key": record.env_key,
        "is_active": record.is_active,
        "has_key": has_key,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


async def _probe(provider: Provider, *, timeout_s: float = PROBE_TIMEOUT_S) -> None:
    """发起一次最小流式请求，要求至少收到一个分块。"""
    usage = ModelUsage()

    async def run() -> None:
        async for chunk in provider.stream(
            system="You are a connectivity probe.",
            messages=[Msg.user("ping")],
            tools=[],
            usage=usage,
        ):
            if chunk.event in {StreamEvent.TEXT_DELTA, StreamEvent.MESSAGE_END}:
                return

    async with asyncio.timeout(timeout_s):
        await run()


def _preset_timeouts(settings: Settings, kind: str) -> tuple[float, float]:
    for preset in settings.providers.values():
        if preset.kind == kind:
            return preset.first_byte_timeout_s, preset.idle_timeout_s
    return 30.0, 60.0


def create_config_router(
    *,
    store_factory: Callable[[], ConfigStore],
    resolver: ProviderResolver | None,
    settings: Settings,
    probe_client_factory: Callable[[float, float], httpx.AsyncClient] | None = None,
    require_user: Callable | None = None,
) -> APIRouter:
    """构建 provider 配置的 API 路由；所有配置按当前用户隔离。"""
    from fastapi import Depends

    router = APIRouter(prefix="/v1/config")

    def store() -> ConfigStore:
        return store_factory()

    def _has_effective_key(record) -> bool:
        if record.kind == "fake":
            return True
        return store().resolve_key(record.id, user_id=record.user_id) is not None

    @router.get("/presets")
    async def list_presets(user=Depends(require_user) if require_user else None) -> dict[str, list[dict[str, Any]]]:
        # fake 不作为预设提供：它没有可填的 base_url/key，一键填入毫无内容，
        # 反而让人误以为"选了就能用"。离线运行走 model.provider="fake" 的显式
        # 选择，或协议类型下拉里的"离线 Fake"——那是类型，不是预设。
        presets = [
            {
                "name": name,
                "kind": preset.kind,
                "base_url": preset.base_url,
                "env_key": preset.env_key,
                "has_env_key": bool(preset.env_key and os.getenv(preset.env_key)),
            }
            for name, preset in settings.providers.items()
            if preset.kind != "fake"
        ]
        return {"presets": presets}

    @router.get("")
    async def get_config(user: CurrentUser = Depends(require_user) if require_user else None) -> dict[str, Any]:
        user_id = user.id if user is not None else ""
        records = store().list_configs(user_id=user_id)
        active = store().get_active(user_id=user_id)
        return {
            "configured": active is not None,
            "active_id": active.id if active is not None else None,
            "configs": [
                _record_payload(record, has_key=_has_effective_key(record))
                for record in records
            ],
        }

    @router.post("")
    async def create_config(
        payload: ConfigPayload, user: CurrentUser = Depends(require_user) if require_user else None
    ) -> dict[str, Any]:
        """校验并新建一份 provider 配置，成功后失效解析器缓存。"""
        user_id = user.id if user is not None else ""
        errors = _field_errors(payload, settings)
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        if payload.kind != "fake" and not _effective_secret(store(), payload, config_id=None):
            raise HTTPException(status_code=422, detail={"errors": [_missing_secret_error()]})
        try:
            record = await asyncio.to_thread(
                store().create,
                name=payload.name.strip(),
                kind=payload.kind,
                base_url=payload.base_url,
                model=payload.model.strip(),
                env_key=payload.env_key,
                api_key=payload.api_key,
                activate=payload.activate,
                user_id=user_id,
            )
        except DuplicateConfigName as exc:
            raise HTTPException(
                status_code=409, detail={"errors": [{"field": "name", "message": str(exc)}]}
            ) from exc
        if resolver is not None:
            resolver.invalidate()
        return {"config": _record_payload(record, has_key=_has_effective_key(record))}

    @router.put("/{config_id}")
    async def update_config(
        config_id: int, payload: ConfigPayload, user: CurrentUser = Depends(require_user) if require_user else None
    ) -> dict[str, Any]:
        """更新指定配置；保留未重新提供的已存密钥并失效解析器缓存。"""
        user_id = user.id if user is not None else ""
        errors = _field_errors(payload, settings)
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        if store().get(config_id, user_id=user_id) is None:
            raise HTTPException(status_code=404, detail="配置不存在")
        if payload.kind != "fake" and not _effective_secret(
            store(), payload, config_id=config_id, user_id=user_id
        ):
            raise HTTPException(status_code=422, detail={"errors": [_missing_secret_error()]})
        try:
            record = await asyncio.to_thread(
                store().update,
                config_id,
                name=payload.name.strip(),
                kind=payload.kind,
                base_url=payload.base_url,
                model=payload.model.strip(),
                env_key=payload.env_key,
                api_key=payload.api_key,
                user_id=user_id,
            )
        except DuplicateConfigName as exc:
            raise HTTPException(
                status_code=409, detail={"errors": [{"field": "name", "message": str(exc)}]}
            ) from exc
        if resolver is not None:
            resolver.invalidate()
        return {"config": _record_payload(record, has_key=_has_effective_key(record))}

    @router.post("/{config_id}/activate")
    async def activate_config(
        config_id: int, user: CurrentUser = Depends(require_user) if require_user else None
    ) -> dict[str, Any]:
        """激活指定配置，使其成为该用户解析器当前使用的 provider。"""
        user_id = user.id if user is not None else ""
        existing = store().get(config_id, user_id=user_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="配置不存在")
        url_error = validate_user_provider_url(existing.base_url, existing.kind, settings)
        if url_error is not None:
            raise HTTPException(
                status_code=422,
                detail={"errors": [{"field": "base_url", "message": url_error}]},
            )
        try:
            record = await asyncio.to_thread(store().activate, config_id, user_id=user_id)
        except ConfigNotFound as exc:
            raise HTTPException(status_code=404, detail="配置不存在") from exc
        if resolver is not None:
            resolver.invalidate()
        return {"config": _record_payload(record, has_key=_has_effective_key(record))}

    @router.delete("/{config_id}")
    async def delete_config(
        config_id: int, user: CurrentUser = Depends(require_user) if require_user else None
    ) -> dict[str, bool]:
        """删除指定配置；拒绝删除当前激活项。"""
        user_id = user.id if user is not None else ""
        try:
            await asyncio.to_thread(store().delete, config_id, user_id=user_id)
        except ConfigNotFound as exc:
            raise HTTPException(status_code=404, detail="配置不存在") from exc
        except ActiveConfigDeleteError as exc:
            raise HTTPException(
                status_code=409, detail={"errors": [{"field": "config", "message": str(exc)}]}
            ) from exc
        if resolver is not None:
            resolver.invalidate()
        return {"ok": True}

    @router.post("/probe")
    async def probe_config(
        payload: ProbePayload, user: CurrentUser = Depends(require_user) if require_user else None
    ) -> dict[str, Any]:
        """探测一份配置的连通性，返回是否可用及延迟。"""
        user_id = user.id if user is not None else ""
        errors = _field_errors(payload, settings)
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        if payload.kind == "fake":
            return {"ok": True, "latency_ms": 0, "model": payload.model, "error": None}

        secret = _effective_secret(store(), payload, config_id=payload.config_id, user_id=user_id)
        if not secret:
            return {
                "ok": False, "latency_ms": 0, "model": payload.model, "error": _MISSING_SECRET_MESSAGE,
            }

        first_byte, idle = _preset_timeouts(settings, payload.kind)
        if probe_client_factory is not None:
            client = probe_client_factory(first_byte, idle)
        else:
            client = httpx.AsyncClient(timeout=httpx.Timeout(idle, connect=first_byte))
        started = time.perf_counter()
        try:
            provider = build_provider_from_fields(
                kind=payload.kind,
                base_url=payload.base_url,
                api_key=secret,
                model=payload.model,
                temperature=settings.model.temperature,
                max_tokens=1,
                first_byte_timeout_s=first_byte,
                idle_timeout_s=idle,
                client=client,
            )
            await _probe(provider)
        except (ProviderError, TimeoutError) as exc:
            return {
                "ok": False,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "model": payload.model,
                "error": str(exc) or type(exc).__name__,
            }
        except Exception as exc:  # noqa: BLE001 - 将任何传输失败暴露给 UI
            return {
                "ok": False,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "model": payload.model,
                "error": str(exc) or type(exc).__name__,
            }
        finally:
            await client.aclose()
        return {
            "ok": True,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "model": payload.model,
            "error": None,
        }

    return router
