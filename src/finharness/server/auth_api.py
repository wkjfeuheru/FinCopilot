"""认证相关的 HTTP 路由：注册、登录、登出、当前用户。

令牌通过 HttpOnly Cookie 下发（同源浏览器无需任何请求头改造），
同时以 JSON 返回一次，供脚本与测试使用 Bearer 通道。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from finharness.auth.dependency import SESSION_COOKIE, resolve_user
from finharness.auth.store import (
    DuplicateUsername,
    IssuedSession,
    UserStore,
    UserStoreError,
)


class RegisterPayload(BaseModel):
    username: str
    password: str


class LoginPayload(BaseModel):
    username: str
    password: str


def _session_response(
    session: IssuedSession, response: Response, *, secure: bool, ttl_s: int
) -> dict[str, Any]:
    response.set_cookie(
        SESSION_COOKIE,
        session.token,
        max_age=ttl_s,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )
    return {
        "token": session.token,
        "user": {"id": session.user.id, "username": session.user.username},
        "expires_at": session.expires_at,
    }


def create_auth_router(
    *,
    store: UserStore,
    ttl_s: int,
    secure_cookie: bool,
    allow_register: bool = True,
    claim_legacy=None,
) -> APIRouter:
    """构建认证路由；``claim_legacy`` 见 ``UserStore.register``。"""
    router = APIRouter(prefix="/v1/auth")

    @router.post("/register")
    async def register(payload: RegisterPayload, response: Response) -> dict[str, Any]:
        if not allow_register:
            raise HTTPException(status_code=403, detail="注册已关闭")
        try:
            session = store.register(
                payload.username.strip(),
                payload.password,
                ttl_s=ttl_s,
                claim_legacy=claim_legacy,
            )
        except DuplicateUsername as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UserStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _session_response(session, response, secure=secure_cookie, ttl_s=ttl_s)

    @router.post("/login")
    async def login(payload: LoginPayload, response: Response) -> dict[str, Any]:
        from finharness.auth.store import InvalidCredentials

        try:
            session = store.login(payload.username.strip(), payload.password, ttl_s=ttl_s)
        except InvalidCredentials as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except UserStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _session_response(session, response, secure=secure_cookie, ttl_s=ttl_s)

    @router.post("/logout")
    async def logout(request: Request, response: Response) -> dict[str, Any]:
        token = request.cookies.get(SESSION_COOKIE, "")
        if not token:
            from finharness.auth.dependency import _bearer_token

            token = _bearer_token(request)
        revoked = store.revoke(token) if token else False
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": revoked}

    @router.get("/me")
    async def me(request: Request) -> dict[str, Any]:
        user = resolve_user(request, store)
        if user is None:
            raise HTTPException(status_code=401, detail="未登录或会话已过期")
        return {"user": {"id": user.id, "username": user.username}}

    return router
