"""FastAPI 认证依赖：从 Cookie 或 Bearer 解析当前用户。

支持双通道，二者顺序为先 Cookie 后 Bearer（同源浏览器走 Cookie，
脚本与测试用 ``Authorization: Bearer``）。失败统一 401，
错误信息不区分"未登录"与"会话过期"，避免探测。
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from finharness.auth.store import CurrentUser, UserStore

SESSION_COOKIE = "finharness_session"


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def resolve_user(request: Request, store: UserStore) -> CurrentUser | None:
    """从请求中解析用户；无效凭据返回 None 而不抛出。"""
    token = request.cookies.get(SESSION_COOKIE, "") or _bearer_token(request)
    return store.resolve_token(token)


def create_require_user(store: UserStore):
    """构建 ``require_user`` 依赖；store 以闭包方式注入。"""

    def require_user(request: Request) -> CurrentUser:
        user = resolve_user(request, store)
        if user is None:
            raise HTTPException(status_code=401, detail="未登录或会话已过期")
        return user

    return require_user
