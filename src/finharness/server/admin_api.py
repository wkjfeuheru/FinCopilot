"""管理员页的查询 API（/v1/admin）：用户总览与用量汇总。

权限：全部端点走 ``require_admin``（未登录 401 → 非管理员 403），依赖内
含 ``require_user``，无匿名路径。数据合并三源：

* ``UserStore``：账号（用户名/角色/注册时间）——身份的唯一事实源；
* ``MemoryStore``：每用户对话数与最后活跃（conversations 表聚合）；
* ``UsageStore``：每用户轮数与 token（usage_turns 聚合，窗口可过滤）。

纯查看：本期不提供任何管理动作（禁用/改配额），后续需要时在此路由上扩。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from finharness.auth.store import CurrentUser

__all__ = ["create_admin_router"]

# 窗口参数 → 时间偏移；白名单校验拒绝未知值（422），避免任意 seconds 注入。
_WINDOWS: dict[str, timedelta] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}


def _since_from_window(window: str) -> str | None:
    if window == "all":
        return None
    offset = _WINDOWS.get(window)
    if offset is None:
        raise HTTPException(
            status_code=422,
            detail=f"未知时间窗：{window}；可用值 all/24h/7d/30d",
        )
    return (datetime.now(timezone.utc) - offset).isoformat(timespec="milliseconds")


def create_admin_router(
    *,
    user_store,
    memory_store,
    usage_store,
    require_user: Callable,
) -> APIRouter:
    """构建 /v1/admin 路由；依赖 ``require_user`` 叠加管理员判定。"""

    router = APIRouter(prefix="/v1/admin", tags=["admin"])
    require_admin = Depends(_make_admin_dependency(require_user))

    @router.get("/status")
    def status(user: CurrentUser = require_admin) -> dict[str, Any]:
        """前端门控：管理员返回 ``{"is_admin": true}``，其余已被 403 拦下。"""
        return {"is_admin": user.is_admin}

    @router.get("/users")
    def users(
        user: CurrentUser = require_admin,
        window: str = Query(default="all"),
    ) -> dict[str, Any]:
        """用户总览：账号 × 对话数 × 用量（总累计 + 窗口内并列）。"""
        since = _since_from_window(window)
        accounts = user_store.list_users()
        conv_stats = memory_store.conversation_stats_by_user()
        totals = {row["user_id"]: row for row in usage_store.totals_by_user()}
        windowed = {
            row["user_id"]: row for row in usage_store.totals_by_user(since=since)
        }

        rows = []
        for account in accounts:
            user_id = account["id"]
            total = totals.get(user_id, {})
            win = windowed.get(user_id, {})
            stats = conv_stats.get(user_id, {})
            rows.append(
                {
                    **account,
                    "conversations": stats.get("conversations", 0),
                    "last_active_at": stats.get("last_active_at", ""),
                    "turns": int(total.get("turns", 0) or 0),
                    "input_tokens": int(total.get("input_tokens", 0) or 0),
                    "output_tokens": int(total.get("output_tokens", 0) or 0),
                    "cache_hit_tokens": int(total.get("cache_hit_tokens", 0) or 0),
                    "window_turns": int(win.get("turns", 0) or 0),
                    "window_input_tokens": int(win.get("input_tokens", 0) or 0),
                    "window_output_tokens": int(win.get("output_tokens", 0) or 0),
                }
            )
        return {"users": rows, "window": window}

    @router.get("/usage/summary")
    def usage_summary(
        user: CurrentUser = require_admin,
        window: str = Query(default="all"),
    ) -> dict[str, Any]:
        """汇总卡片：总用户数 + 窗口内活跃/轮数/token。"""
        since = _since_from_window(window)
        return {
            "window": window,
            "total_users": user_store.count_users(),
            **usage_store.summary(since=since),
        }

    return router


def _make_admin_dependency(require_user: Callable):
    """把通用 require_admin 依赖的构建收敛到一处（供 Depends 使用）。"""
    from finharness.auth.dependency import create_require_admin

    return create_require_admin(require_user)
