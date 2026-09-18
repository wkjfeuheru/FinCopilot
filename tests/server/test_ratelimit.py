"""限速与每租户配额：认证端点防滥用、对话端点保公平（隔离方案 P0-5/P0-6）。

这些测试断言的是**默认配置**下的行为，而不是某个可选开关打开后的行为：
默认值本身就是防线的一部分，如果默认不限速，那这条防线在实际部署里就不存在。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from finharness.auth.ratelimit import RateLimiter
from finharness.config.settings import Settings
from finharness.provider.fake import FakeProvider
from finharness.server.api import create_app


def make_client(tmp_path, **overrides) -> TestClient:
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
        **overrides,
    )
    return TestClient(create_app(FakeProvider(["ok"]), settings=settings))


# -- RateLimiter 单元行为 ------------------------------------------------------

def test_limiter_allows_up_to_the_limit_then_rejects():
    limiter = RateLimiter(limit=3, window_s=60)

    verdicts = [limiter.check("k").allowed for _ in range(4)]

    assert verdicts == [True, True, True, False]


def test_a_rejected_call_reports_how_long_to_wait():
    limiter = RateLimiter(limit=1, window_s=60)
    limiter.check("k")

    decision = limiter.check("k")

    assert decision.allowed is False
    assert decision.retry_after_s >= 1


def test_keys_are_counted_independently():
    limiter = RateLimiter(limit=1, window_s=60)
    limiter.check("a")

    assert limiter.check("a").allowed is False
    assert limiter.check("b").allowed is True


def test_a_zero_limit_disables_limiting():
    limiter = RateLimiter(limit=0, window_s=60)

    assert limiter.enabled is False
    assert all(limiter.check("k").allowed for _ in range(50))


def test_reset_clears_a_key():
    limiter = RateLimiter(limit=1, window_s=60)
    limiter.check("k")

    limiter.reset("k")

    assert limiter.check("k").allowed is True


def test_an_expired_window_rolls_over():
    limiter = RateLimiter(limit=1, window_s=0.05)
    limiter.check("k")
    assert limiter.check("k").allowed is False

    import time

    time.sleep(0.06)

    assert limiter.check("k").allowed is True


def test_counters_are_bounded():
    """键来自 IP/用户名，因此计数表本身不能无界增长。"""
    limiter = RateLimiter(limit=5, window_s=60, max_keys=10)

    for index in range(100):
        limiter.check(f"key-{index}")

    assert len(limiter._windows) <= 10


# -- 登录限速 -----------------------------------------------------------------

def test_repeated_failed_logins_are_throttled(tmp_path):
    client = make_client(tmp_path, auth={"login_max_attempts": 3})
    client.post("/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"})

    statuses = [
        client.post(
            "/v1/auth/login", json={"username": "alice", "password": "wrong-pass-1"}
        ).status_code
        for _ in range(5)
    ]

    assert statuses[:3] == [401, 401, 401]
    assert statuses[3] == 429, "超过上限后必须被限速，而不是继续放行猜测"
    assert statuses[4] == 429


def test_a_successful_login_does_not_consume_the_quota(tmp_path):
    """一次误输入不该占额度：登录成功即清空该键。"""
    client = make_client(tmp_path, auth={"login_max_attempts": 2})
    client.post("/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"})
    client.post("/v1/auth/login", json={"username": "alice", "password": "wrong-pass-1"})

    ok = client.post("/v1/auth/login", json={"username": "alice", "password": "secret-pass-1"})
    again = client.post("/v1/auth/login", json={"username": "alice", "password": "secret-pass-1"})

    assert ok.status_code == 200
    assert again.status_code == 200, "成功登录后不应立刻被自己的额度挡住"


def test_throttling_does_not_leak_whether_a_user_exists(tmp_path):
    """限速是按 (地址, 用户名) 计数的，因此两条路径同样被限。"""
    client = make_client(tmp_path, auth={"login_max_attempts": 1})

    for _ in range(3):
        client.post("/v1/auth/login", json={"username": "ghost", "password": "whatever-1"})
    unknown = client.post(
        "/v1/auth/login", json={"username": "ghost", "password": "whatever-1"}
    )

    assert unknown.status_code == 429


# -- 注册限速 -----------------------------------------------------------------

def test_registration_is_rate_limited(tmp_path):
    client = make_client(tmp_path, auth={"register_max_attempts": 2})

    statuses = [
        client.post(
            "/v1/auth/register", json={"username": f"user{index}", "password": "secret-pass-1"}
        ).status_code
        for index in range(4)
    ]

    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429
    assert statuses[3] == 429


# -- 存量数据继承默认关闭 ------------------------------------------------------

def test_first_registration_does_not_inherit_legacy_data_by_default(tmp_path):
    """默认不允许"谁先注册谁拿到全部历史数据"这一隐式授权。"""
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
    )
    assert settings.auth.claim_legacy_on_first_register is False

    app = create_app(FakeProvider(["ok"]), settings=settings)
    client = TestClient(app)
    # 造一条 user_id='' 的存量对话，它属于"单用户时代"。
    app.state.memory_store.ensure_conversation("c_legacy", user_id="")

    client.post("/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"})
    user_id = app.state.user_store.find_by_username("alice").id

    assert app.state.memory_store.get_conversation("c_legacy", user_id=user_id) is None
    # 而它仍然存在（没有被删除），只是不属于新用户。
    assert app.state.memory_store.get_conversation("c_legacy", user_id="") is not None


def test_legacy_inheritance_can_be_opted_into_explicitly(tmp_path):
    """显式打开时，首注册者仍可认领存量数据（单用户升级路径）。"""
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
        auth={"claim_legacy_on_first_register": True},
    )
    app = create_app(FakeProvider(["ok"]), settings=settings)
    client = TestClient(app)
    app.state.memory_store.ensure_conversation("c_legacy", user_id="")

    client.post("/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"})
    user_id = app.state.user_store.find_by_username("alice").id

    assert app.state.memory_store.get_conversation("c_legacy", user_id=user_id) is not None


# -- 每租户对话配额 ------------------------------------------------------------

def _two_users(tmp_path, **overrides):
    """同一应用上的两个独立客户端与各自凭据。

    必须是两个 ``TestClient``：``resolve_user`` 先看 Cookie 再看 Bearer，而一个
    TestClient 共用一个 Cookie 罐，于是第二个用户登录后第一个用户的请求会被认成
    第二个用户——那恰好是"配额按用户隔离"这条断言要排除的情况。
    """
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
        **overrides,
    )
    app = create_app(FakeProvider(["ok"]), settings=settings)

    def login(username: str):
        client = TestClient(app)
        for path in ("register", "login"):
            response = client.post(
                f"/v1/auth/{path}",
                json={"username": username, "password": "secret-pass-1"},
            )
        return client, {"Authorization": f"Bearer {response.json()['token']}"}

    return login("alice"), login("bob"), app


def test_a_tenant_turn_budget_is_enforced(tmp_path):
    """低额度配置下，超出的轮次返回 429；另一个租户不受影响。"""
    (alice_client, alice), (bob_client, bob), _ = _two_users(
        tmp_path, quota={"turns_per_window": 2, "window_s": 3600}
    )

    statuses = [
        alice_client.post(
            "/v1/chat/stream", json={"message": "q"}, headers=alice
        ).status_code
        for _ in range(3)
    ]
    other = bob_client.post(
        "/v1/chat/stream", json={"message": "q"}, headers=bob
    ).status_code

    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429, "本时段额度用尽后必须拒绝"
    assert other == 200, "配额按租户独立，不牵连他人"


def test_quota_rejection_carries_retry_after(tmp_path):
    (client, headers), _, _ = _two_users(
        tmp_path, quota={"turns_per_window": 1, "window_s": 3600}
    )
    client.post("/v1/chat/stream", json={"message": "q"}, headers=headers)

    blocked = client.post("/v1/chat/stream", json={"message": "q"}, headers=headers)

    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1


def test_concurrent_stream_cap_is_enforced(tmp_path):
    """同一租户同时进行的流有上限；被占满时新的流被拒。"""
    (client, headers), _, app = _two_users(
        tmp_path, quota={"turns_per_window": 0, "max_concurrent_streams": 1}
    )
    registry = app.state.session_registry
    alice_id = app.state.user_store.find_by_username("alice").id

    # 直接构造"该租户已有一个在飞的流"，避免依赖并发时序。
    import asyncio

    async def occupy():
        session = await registry.ensure(None, user_id=alice_id)
        registry.mark_busy(session)

    asyncio.run(occupy())

    blocked = client.post("/v1/chat/stream", json={"message": "q"}, headers=headers)

    assert blocked.status_code == 429
    assert "同时进行" in blocked.text


def test_concurrent_stream_cap_is_off_by_default(tmp_path):
    """默认不限并发：这是刻意的，因为泄漏的 busy 会话会锁死整个租户。

    SSE 生成器若未被消费完，``release`` 走不到，busy 标记要等回收窗口
    （默认 1 小时）才消失。此时若并发上限默认生效，几条泄漏就能让一个租户
    被拒一小时——比它要防的滥用更糟。因此默认关闭，并由本用例守住这一点。
    """
    settings = Settings(
        data={"cache_dir": tmp_path / "cache"},
        paths={
            "output_dir": tmp_path / "output",
            "memory_db": tmp_path / "state" / "memory.db",
            "auth_db": tmp_path / "state" / "users.db",
        },
    )
    assert settings.quota.max_concurrent_streams == 0

    app = create_app(FakeProvider(["ok"]), settings=settings)
    client = TestClient(app)
    client.post("/v1/auth/register", json={"username": "alice", "password": "secret-pass-1"})
    token = client.post(
        "/v1/auth/login", json={"username": "alice", "password": "secret-pass-1"}
    ).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    registry = app.state.session_registry
    alice_id = app.state.user_store.find_by_username("alice").id

    import asyncio

    async def leak_many():
        for _ in range(20):
            session = await registry.ensure(None, user_id=alice_id)
            registry.mark_busy(session)

    asyncio.run(leak_many())

    response = client.post("/v1/chat/stream", json={"message": "q"}, headers=headers)

    assert response.status_code == 200, "默认不得因泄漏的 busy 会话而拒绝该租户"
