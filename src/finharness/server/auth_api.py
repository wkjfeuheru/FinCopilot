"""认证相关的 HTTP 路由：注册、登录、登出、当前用户。

令牌通过 HttpOnly Cookie 下发（同源浏览器无需任何请求头改造），
同时以 JSON 返回一次，供脚本与测试使用 Bearer 通道。

注册与登录都受限速保护：前者防批量注册，后者防密码暴力猜测。计数键按
**客户端地址**（登录再加上归一化后的用户名），使针对单个账号的猜测无法
靠切换源地址稀释，同时一个地址上的正常用户不会被别人的失败拖累。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from finharness.auth.dependency import SESSION_COOKIE, resolve_user
from finharness.auth.ratelimit import RateLimitDecision, RateLimiter
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


def client_key(request: Request) -> str:
    """用于限速的客户端标识。

    直接取 ``request.client.host``，**不**解析 ``X-Forwarded-For``：该头由
    客户端自由伪造，信任它等于让攻击者用任意字符串把自己拆成无数个键，
    反而绕过限速。真实部署应在反向代理层限速，或由代理覆写并保证该头可信
    后再于此处读取（见隔离方案 Phase 2）。
    """
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or "unknown"


def _reject_limited(decision: RateLimitDecision, what: str) -> None:
    raise HTTPException(
        status_code=429,
        detail=f"{what}过于频繁，请 {decision.retry_after_s} 秒后重试",
        headers={"Retry-After": str(decision.retry_after_s)},
    )


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
        "user": {
            "id": session.user.id,
            "username": session.user.username,
            "role": session.user.role,
        },
        "expires_at": session.expires_at,
    }


def create_auth_router(
    *,
    store: UserStore,
    ttl_s: int,
    secure_cookie: bool,
    allow_register: bool = True,
    claim_legacy=None,
    bootstrap_admin: bool = False,
    login_limiter: RateLimiter | None = None,
    register_limiter: RateLimiter | None = None,
) -> APIRouter:
    """构建认证路由；``claim_legacy`` 见 ``UserStore.register``。

    ``bootstrap_admin``：开启期间注册的用户 role='admin'（管理员引导，
    见 ``AuthSettings.admin_bootstrap``）。

    ``login_limiter`` / ``register_limiter`` 省略时该端点不限速（保持单用户
    本地部署的既有行为）；服务端按 settings 显式注入。
    """
    router = APIRouter(prefix="/v1/auth")

    @router.post("/register")
    async def register(
        payload: RegisterPayload, request: Request, response: Response
    ) -> dict[str, Any]:
        if not allow_register:
            raise HTTPException(status_code=403, detail="注册已关闭")
        if register_limiter is not None:
            decision = register_limiter.check(f"register:{client_key(request)}")
            if not decision.allowed:
                _reject_limited(decision, "注册")
        try:
            session = store.register(
                payload.username.strip(),
                payload.password,
                ttl_s=ttl_s,
                claim_legacy=claim_legacy,
                bootstrap_admin=bootstrap_admin,
            )
        except DuplicateUsername as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UserStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _session_response(session, response, secure=secure_cookie, ttl_s=ttl_s)

    @router.post("/login")
    async def login(
        payload: LoginPayload, request: Request, response: Response
    ) -> dict[str, Any]:
        from finharness.auth.store import InvalidCredentials

        username = payload.username.strip()
        # 键同时含源地址与用户名：只按地址会让同一代理后的所有用户互相挤占，
        # 只按用户名则攻击者可轮换用户名绕过。归一化大小写，避免用大小写变体
        # 把同一个账号拆成多个计数槽。
        limiter_key = f"login:{client_key(request)}:{username.lower()}"
        if login_limiter is not None:
            decision = login_limiter.check(limiter_key)
            if not decision.allowed:
                _reject_limited(decision, "登录尝试")
        try:
            session = store.login(username, payload.password, ttl_s=ttl_s)
        except InvalidCredentials as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except UserStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # 成功后清空该键，使一次误输入不会占用后续额度。
        if login_limiter is not None:
            login_limiter.reset(limiter_key)
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
        return {"user": {"id": user.id, "username": user.username, "role": user.role}}

    return router
