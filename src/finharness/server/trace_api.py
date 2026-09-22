"""监控平台的查询 API（docs 03.14.4）。

三个只读数据端点（run 列表 / run 完整 trace / 指标聚合）加一个状态端点。

权限与可用性是**两个正交的状态**，必须能被前端区分，否则运维方只会看到
一个无信息量的 404：

* 未登录 → 401（由 ``require_user`` 负责）；
* 已登录但不在 ``observability.trace_store.admin_users`` 白名单 → 403；
* 在名单内但 ``trace_store.enabled=false`` → **503**，并给出开启方法。

因此路由器**始终挂载**（未启用时数据端点返回 503 而非 404），另有
``/v1/trace/status`` 供前端在渲染前判定该显示数据、引导开启、还是无权限。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from finharness.auth.store import CurrentUser

__all__ = ["create_trace_router"]

_DISABLED_HINT = (
    "运行监控未启用。请在 settings.json 中设置 "
    "observability.trace_store.enabled=true，并把需要查看监控的用户名加入 "
    "observability.trace_store.admin_users，然后重启服务。"
)


def create_trace_router(
    trace_store: Any | None,
    admin_users: list[str],
    require_user: Callable | None = None,
) -> APIRouter:
    """构建 /v1/trace 路由；``trace_store`` 为 None 表示监控未启用。"""

    router = APIRouter(prefix="/v1/trace", tags=["trace"])
    dep = Depends(require_user) if require_user else None
    admins = set(admin_users or [])

    def _is_admin(user: CurrentUser) -> bool:
        return user.username in admins

    def _require_admin(user: CurrentUser) -> None:
        if not _is_admin(user):
            raise HTTPException(status_code=403, detail="无监控访问权限")

    def _require_enabled() -> None:
        if trace_store is None:
            raise HTTPException(status_code=503, detail=_DISABLED_HINT)

    @router.get("/status")
    def status(user: CurrentUser = dep) -> dict[str, Any]:
        """监控可用性：前端据此决定渲染数据、引导开启还是提示无权限。

        ``admin_users`` 为空是一个常见的配置疏漏（开了监控但没人能看），
        因此单独回报 ``admin_configured``，让前端提示补上白名单。
        """
        return {
            "enabled": trace_store is not None,
            "is_admin": _is_admin(user),
            "admin_configured": bool(admins),
        }

    @router.get("/runs")
    def list_runs(
        user: CurrentUser = dep,
        source: str | None = Query(default=None),
        status_filter: str | None = Query(default=None, alias="status"),
        user_id: str | None = Query(default=None),
        conversation_id: str | None = Query(default=None),
        since: str | None = Query(default=None),
        until: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        _require_admin(user)
        _require_enabled()
        filters = {
            "source": source,
            "status": status_filter,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "since": since,
            "until": until,
        }
        runs = trace_store.list_runs(**filters, limit=limit, offset=offset)
        total = trace_store.count_runs(**filters)
        return {"runs": runs, "total": total, "limit": limit, "offset": offset}

    @router.get("/runs/{run_id}")
    def run_detail(run_id: str, user: CurrentUser = dep) -> dict[str, Any]:
        _require_admin(user)
        _require_enabled()
        detail = trace_store.run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="运行不存在")
        return detail

    @router.get("/metrics")
    def metrics(
        user: CurrentUser = dep,
        source: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        since: str | None = Query(default=None),
        until: str | None = Query(default=None),
    ) -> dict[str, Any]:
        _require_admin(user)
        _require_enabled()
        return trace_store.metrics_summary(
            source=source, user_id=user_id, since=since, until=until
        )

    return router
