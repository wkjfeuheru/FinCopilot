"""旧布局状态数据的检测与搬迁（隔离方案 P0-3 的配套迁移）。

覆盖三件事：**检测**（旧数据还在而当前位置没有 → 启动即失败，而不是在空库上
静默运行）、**搬迁**（含密钥与配置库的顺序与配对）、**边界**（目标已有数据、
密钥冲突、重复执行、显式指回旧路径）。

这里的断言都直接检查**数据可用性**（账号能否登录、供应商 Key 能否解出），
而不是只比较文件是否拷过去了：文件到位但解不开，正是搬迁最典型的失败形态。
"""

from __future__ import annotations

import pathlib

import pytest

from finharness.auth.store import UserStore
from finharness.config import state_migration as sm
from finharness.config.crypto import SecretCipher
from finharness.config.settings import Settings, SettingsError
from finharness.config.store import ConfigStore
from finharness.context.memory.store import MemoryStore

PASSWORD = "secret-pass-1"


def make_settings(root: pathlib.Path, **path_overrides) -> Settings:
    """指向 root 下的旧/新两套位置。"""
    paths = {
        "output_dir": root / "output",
        "state_dir": root / "state",
        "memory_db": root / "state" / "memory.db",
        "auth_db": root / "state" / "users.db",
        "config_db": root / "state" / "config.db",
        "secret_key": root / "state" / "secret.key",
    }
    paths.update(path_overrides)
    return Settings(paths=paths, data={"cache_dir": root / "data_cache"})


def build_legacy(root: pathlib.Path, *, users=("alice", "bob"), provider=True):
    """在旧布局（data_cache/）下造出与升级前一致的存量数据。"""
    legacy = root / "data_cache"
    legacy.mkdir(parents=True, exist_ok=True)
    store = UserStore(legacy / "users.db")
    for name in users:
        store.register(name, PASSWORD)
    MemoryStore(legacy / "memory.db").ensure_conversation("c_old", user_id="u_legacy")
    if provider:
        configs = ConfigStore(legacy / "config.db", cipher=SecretCipher(legacy / "secret.key"))
        configs.create(
            user_id="",
            name="deepseek",
            kind="openai_compat",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            env_key="DEEPSEEK_API_KEY",
            api_key="sk-REAL-SECRET",
            activate=True,
        )
    return legacy


def build_destination(root: pathlib.Path, *, users=(), own_key=False):
    """在当前位置预先放一些无关数据（模拟被测试污染过的 state/）。"""
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    if users:
        store = UserStore(state / "users.db")
        for name in users:
            store.register(name, PASSWORD)
    if own_key:
        SecretCipher(state / "secret.key")  # 生成一把与旧布局不同的主密钥
    return state


# -- 检测 ---------------------------------------------------------------------

def test_missing_accounts_are_detected(tmp_path):
    build_legacy(tmp_path)

    state = sm.detect_legacy_state(make_settings(tmp_path))

    assert state.needs_migration is True
    assert state.missing_accounts == {"alice", "bob"}


def test_no_legacy_data_means_nothing_to_migrate(tmp_path):
    state = sm.detect_legacy_state(make_settings(tmp_path))

    assert state.needs_migration is False


def test_a_migrated_layout_is_not_flagged_again(tmp_path):
    """搬迁完成后再检测应放行，否则守卫会永远拦住启动。"""
    build_legacy(tmp_path)
    settings = make_settings(tmp_path)
    sm.migrate_legacy_state(settings)

    state = sm.detect_legacy_state(settings)

    assert state.needs_migration is False


def test_pointing_state_files_back_at_the_legacy_dir_is_rejected(tmp_path):
    """``paths.*`` 指回 ``data_cache/`` **不是**逃生通道，加载期就拒绝。

    这曾经是守卫提示里的一个建议，但它是错的：状态文件落在 agent 可达目录内
    正是 P0-3 要消除的布局，模型级校验不允许。搬迁是唯一的正规路径。
    """
    legacy = build_legacy(tmp_path)

    with pytest.raises(Exception) as caught:
        make_settings(
            tmp_path,
            state_dir=legacy,
            memory_db=legacy / "memory.db",
            auth_db=legacy / "users.db",
            config_db=legacy / "config.db",
            secret_key=legacy / "secret.key",
        )

    assert "agent 可达目录" in str(caught.value)


def test_startup_fails_instead_of_running_on_an_empty_universe(tmp_path):
    """核心验收：账号在旧布局、当前位置没有 → 启动即失败并指路。"""
    build_legacy(tmp_path)

    with pytest.raises(SettingsError) as caught:
        make_settings(tmp_path).validate_runtime(require_api_key=False)

    message = str(caught.value)
    assert "alice" in message
    assert "state_migration" in message, "错误信息必须给出搬迁命令"


# -- 搬迁 ---------------------------------------------------------------------

def test_migration_makes_the_old_accounts_usable_again(tmp_path):
    build_legacy(tmp_path)
    settings = make_settings(tmp_path)

    sm.migrate_legacy_state(settings)
    settings.validate_runtime(require_api_key=False)

    # 关键验证：搬过来的账号真能登录，而不是仅仅文件存在。
    store = UserStore(tmp_path / "state" / "users.db")
    session = store.login("alice", PASSWORD)
    assert session.user.username == "alice"


def test_migration_carries_the_provider_key_and_it_decrypts(tmp_path):
    """密钥与配置库的配对是搬迁最容易静默出错的地方。"""
    build_legacy(tmp_path)
    settings = make_settings(tmp_path)

    report = sm.migrate_legacy_state(settings)

    assert report.verified is True
    configs = ConfigStore(
        tmp_path / "state" / "config.db",
        cipher=SecretCipher(tmp_path / "state" / "secret.key"),
    )
    record = configs.list_configs(user_id="")[0]
    assert configs.resolve_key(record.id, user_id="") == "sk-REAL-SECRET"


def test_migration_preserves_the_legacy_directory_as_backup(tmp_path):
    legacy = build_legacy(tmp_path)

    sm.migrate_legacy_state(make_settings(tmp_path))

    assert (legacy / "users.db").is_file()
    assert (legacy / "config.db").is_file()
    assert (legacy / "secret.key").is_file()


def test_a_dry_run_writes_nothing(tmp_path):
    build_legacy(tmp_path)
    settings = make_settings(tmp_path)

    report = sm.migrate_legacy_state(settings, dry_run=True)

    assert report.dry_run is True
    assert not (tmp_path / "state" / "users.db").exists()
    assert not (tmp_path / "state" / "secret.key").exists()


def test_migration_is_idempotent(tmp_path):
    """可重复执行：第二次不应报错，也不应改变结论。"""
    build_legacy(tmp_path)
    settings = make_settings(tmp_path)

    first = sm.migrate_legacy_state(settings)
    second = sm.migrate_legacy_state(settings)

    assert first.copied
    assert sm.detect_legacy_state(settings).needs_migration is False
    store = UserStore(tmp_path / "state" / "users.db")
    assert store.login("alice", PASSWORD).user.username == "alice"
    assert second.skipped  # 第二次目标已有数据，应转为跳过/合并而非重复复制


# -- 目标已有数据 -------------------------------------------------------------

def test_existing_users_are_kept_and_old_ones_are_merged_in(tmp_path):
    """目标被测试数据污染过时，旧账号要补进来，已有账号不能被覆盖。"""
    build_legacy(tmp_path)
    build_destination(tmp_path, users=("testuser_1", "testuser_2"))
    settings = make_settings(tmp_path)

    report = sm.migrate_legacy_state(settings)

    assert report.merged_accounts == 2
    store = UserStore(tmp_path / "state" / "users.db")
    names = {row for row in ("alice", "bob", "testuser_1", "testuser_2") if store.find_by_username(row)}
    assert names == {"alice", "bob", "testuser_1", "testuser_2"}
    assert store.login("alice", PASSWORD).user.username == "alice"


def test_a_conflicting_key_does_not_produce_an_unreadable_config_db(tmp_path):
    """密钥冲突时绝不复制配置库——那会搬过去一个解不开的库。

    这是本模块最重要的一条边界：宁可保留旧文件并明确报告，也不要制造
    "文件在但用不了"的状态，那看起来像数据损坏。
    """
    build_legacy(tmp_path)
    build_destination(tmp_path, users=("testuser_1",), own_key=True)
    settings = make_settings(tmp_path)

    report = sm.migrate_legacy_state(settings)

    assert report.unreadable_config is True
    assert "config_db" in report.skipped
    # 没有在目标位置留下一个解不开的配置库。
    assert not (tmp_path / "state" / "config.db").exists()
    # 但账号与历史已经可用。
    store = UserStore(tmp_path / "state" / "users.db")
    assert store.login("alice", PASSWORD).user.username == "alice"
    settings.validate_runtime(require_api_key=False)


def test_the_report_explains_what_to_do_about_a_key_conflict(tmp_path):
    build_legacy(tmp_path)
    build_destination(tmp_path, own_key=True)

    text = sm.migrate_legacy_state(make_settings(tmp_path)).describe()

    assert "secret.key" in text
    assert "重新填写供应商配置" in text, "必须给出可执行的下一步，而不只是报告失败"


def test_migration_refuses_when_there_is_nothing_to_migrate(tmp_path):
    with pytest.raises(sm.MigrationError):
        sm.migrate_legacy_state(make_settings(tmp_path))
