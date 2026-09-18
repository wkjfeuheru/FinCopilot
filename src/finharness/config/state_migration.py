"""旧布局状态数据的检测与搬迁（隔离方案 P0-3 的配套迁移）。

**为什么需要这个模块。** P0-3 把密钥与租户数据库从 ``data_cache/`` 搬到了
``paths.state_dir``（默认 ``state/``），并改了默认路径。对一个已经运行过的部署
来说，这意味着重启后应用会去看一个**空**的新位置：账号登不进去、历史看起来没了、
供应商 Key 需要重配——而它和"全新安装"在外部表现上完全一样。静默地把用户的
数据库换掉比搬家本身危险得多，因此这里提供两件事：

* ``check_for_unmigrated_state()`` —— 启动即失败并指路，而不是空转；
* ``migrate_legacy_state()`` —— 一次性搬迁，**复制而非移动**，原目录保留作备份。

**搬迁顺序有硬要求：``secret.key`` 必须先于 ``config.db``。** 供应商配置库里的
``api_key`` 是用主密钥加密的；只搬数据库不搬密钥（或搬了别处的密钥），打开配置库
时会得到 ``SecretCipherError``，而那时数据已经"搬过去"了，故障看起来像损坏。
``migrate_legacy_state`` 因此按此顺序执行，并在结束时真去解一条配置来验证密钥配对。

**不自动执行。** 本模块只被显式调用（启动检查会指路，搬迁本身要运维确认），
不在应用启动时偷偷改动用例数据。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "LegacyState",
    "MigrationReport",
    "check_for_unmigrated_state",
    "detect_legacy_state",
    "migrate_legacy_state",
]

# 旧布局下各状态文件的文件名，相对当时的 ``data.cache_dir``（默认 data_cache/）。
_LEGACY_FILES = {
    "auth_db": "users.db",
    "memory_db": "memory.db",
    "config_db": "config.db",
    "secret_key": "secret.key",
}

# 判断"库里有数据"所用的事实表。选它们是因为一张表中是否有行就是用户能直接
# 感知到的东西：账号数、对话数、供应商配置数。
_COUNT_TABLES = {
    "auth_db": "users",
    "memory_db": "conversations",
    "config_db": "provider_configs",
}


class MigrationError(RuntimeError):
    """搬迁无法安全进行时抛出（绝不半途留下不一致的布局）。"""


def _count_rows(path: Path, table: str) -> int | None:
    """只读统计行数；文件不存在或不是预期的库时返回 None。"""
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def _usernames(path: Path) -> frozenset[str]:
    """只读取出账号名集合（不存在或无法读取时为空集）。

    比较**账号名集合**而不是行数：真正的伤害是"以前能登录的账号现在登不上了"，
    这既包括"新位置为空"，也包括"新位置有数据但完全是另一批"（例如被测试写入
    污染过的库）。只看行数会让后一种情况漏网——两处都非空，但没有任何一个
    账号在两处同时存在。
    """
    if not path.is_file():
        return frozenset()
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return frozenset()
    try:
        rows = connection.execute("SELECT username FROM users").fetchall()
        return frozenset(str(row[0]) for row in rows)
    except sqlite3.Error:
        return frozenset()
    finally:
        connection.close()


def _path_for(settings, key: str) -> Path:
    return Path(getattr(settings.paths, key))


def legacy_state_paths(settings) -> dict[str, Path]:
    """各状态文件在**旧布局**下的位置。

    用当前配置的 ``data.cache_dir`` 反推，而不是写死 ``data_cache/``：旧版本里
    这些文件的默认值是 ``data_cache/<name>``，也就是 ``data.cache_dir/<name>``，
    因此对自定义过 cache 目录的部署同样成立。
    """
    cache_dir = Path(settings.data.cache_dir)
    return {key: cache_dir / name for key, name in _LEGACY_FILES.items()}


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """一处位置上各状态文件的大小与行数。"""

    location: Path
    rows: dict[str, int | None] = field(default_factory=dict)
    present: dict[str, bool] = field(default_factory=dict)

    def count(self, key: str) -> int:
        return int(self.rows.get(key) or 0)

    def has_any_data(self) -> bool:
        return any(self.count(key) > 0 for key in _COUNT_TABLES)

    def describe(self) -> str:
        parts = []
        for key in ("auth_db", "memory_db", "config_db"):
            label, unit = {
                "auth_db": ("账号", "个"),
                "memory_db": ("对话", "条"),
                "config_db": ("供应商配置", "条"),
            }[key]
            parts.append(f"{label} {self.count(key)} {unit}")
        return "、".join(parts)


@dataclass(frozen=True, slots=True)
class LegacyState:
    """一次检测的结论。"""

    legacy: StateSnapshot
    configured: StateSnapshot
    # 旧布局里存在、而当前位置没有的账号。这是"搬迁未完成"最精确的信号。
    missing_accounts: frozenset[str] = frozenset()

    @property
    def needs_migration(self) -> bool:
        """旧布局里有账号在当前库中找不到——重启后这些人就登不进去了。"""
        return bool(self.missing_accounts)

    def guidance(self) -> str:
        legacy_dir = self.legacy.location
        sample = "、".join(sorted(self.missing_accounts)[:5])
        more = "" if len(self.missing_accounts) <= 5 else f" 等 {len(self.missing_accounts)} 个"
        return (
            f"检测到 {len(self.missing_accounts)} 个账号只存在于旧布局 {legacy_dir}"
            f"（{self.legacy.describe()}），当前配置指向的位置 "
            f"{self.configured.location} 里没有它们（{self.configured.describe()}）：{sample}{more}。\n"
            "这会表现为：这些账号无法登录、历史为空、供应商配置需要重配。\n"
            "请二选一：\n"
            "  1) 搬迁（推荐，复制而非移动，原目录保留作备份）：\n"
            "     python -m finharness.config.state_migration --settings <settings.json>\n"
            "  2) 确认要全新开始：把旧目录改名或删除后重启。\n"
            "注意：把 paths.* 指回旧目录不是选项——状态文件位于 data_cache/ 内会被"
            "加载期校验直接拒绝（那正是本次搬家要消除的越权布局）。"
        )


def _snapshot(settings, base: dict[str, Path]) -> StateSnapshot:
    rows: dict[str, int | None] = {}
    present: dict[str, bool] = {}
    for key, path in base.items():
        present[key] = path.is_file()
        table = _COUNT_TABLES.get(key)
        rows[key] = _count_rows(path, table) if table else None
    # 目录仅用于展示与提示：取 auth_db 的父目录即可代表这一组文件的位置。
    return StateSnapshot(location=Path(base["auth_db"]).parent, rows=rows, present=present)


def detect_legacy_state(settings) -> LegacyState:
    """比对旧布局与当前位置，判断是否需要搬迁。"""
    legacy_paths = legacy_state_paths(settings)
    configured_paths = {key: _path_for(settings, key) for key in _LEGACY_FILES}
    legacy_names = _usernames(legacy_paths["auth_db"])
    configured_names = _usernames(configured_paths["auth_db"])
    return LegacyState(
        legacy=_snapshot(settings, legacy_paths),
        configured=_snapshot(settings, configured_paths),
        missing_accounts=frozenset(legacy_names - configured_names),
    )


def check_for_unmigrated_state(settings) -> None:
    """启动检查：旧数据还在、新位置为空则失败并指路。

    刻意做成"失败"而不是"警告 + 继续"：继续的结果是一个看起来正常的空实例，
    运维方要过很久才会发现账号和历史"丢了"，而那时很难联想到是搬家未完成。
    """
    from finharness.config.settings import SettingsError

    state = detect_legacy_state(settings)
    if state.needs_migration:
        raise SettingsError(state.guidance())


# -- 搬迁 ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MigrationReport:
    copied: tuple[str, ...]
    skipped: tuple[str, ...]
    verified: bool
    dry_run: bool = False
    merged_accounts: int = 0
    # 配置库存在但无法用目标位置的密钥解开，因此没有搬运（见 migrate 内注释）。
    unreadable_config: bool = False

    def describe(self) -> str:
        verb = "将复制" if self.dry_run else "已复制"
        lines = [f"{verb}：{', '.join(self.copied) or '（无）'}"]
        if self.merged_accounts:
            lines.append(f"合并进目标库的账号：{self.merged_accounts} 个")
        if self.skipped:
            lines.append(f"跳过：{', '.join(self.skipped)}")
        if self.unreadable_config:
            lines.append(
                "  配置库未搬运：目标位置的 secret.key 解不开它（两处密钥不同）。\n"
                "  账号与历史已就位；供应商配置需要二选一：\n"
                "    a) 改用旧密钥：把 paths.secret_key 指回旧目录的 secret.key；\n"
                "    b) 保持现密钥，重新填写供应商配置。"
            )
        elif "secret_key" in self.skipped:
            lines.append(
                "  注意：两处都有密钥文件，未覆盖。若旧配置库读不出，"
                "请人工确认应使用哪一个密钥。"
            )
        if not self.dry_run:
            lines.append(f"密钥配对校验：{'通过' if self.verified else '未通过'}")
        return "\n".join(lines)


def _copy_sqlite(src: Path, dst: Path) -> None:
    """用 SQLite 备份 API 复制，得到一致快照。

    直接 ``shutil.copy`` 一个正在被使用的库会漏掉 WAL 里尚未合并的事务，
    复制出的是一个丢数据的库。备份 API 由引擎保证一致性。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def _copy_secret(src: Path, dst: Path) -> None:
    """复制主密钥并保持 0600 权限。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = src.read_bytes()
    descriptor = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _destination_is_populated(path: Path, key: str) -> bool:
    """目标位置是否已有数据（用于拒绝覆盖）。"""
    table = _COUNT_TABLES.get(key)
    if table is None:
        # 密钥没有"行数"概念：存在即视为已占用。
        return path.is_file()
    return (_count_rows(path, table) or 0) > 0


def _merge_users(src: Path, dst: Path) -> int:
    """把源库中的账号合并进目标库（只补缺失的用户名，不覆盖已有行）。

    目标库非空时不能直接跳过：那会让"旧布局里的账号"继续缺席，守卫下次启动
    照样失败——等于搬迁没解决问题。合并时按用户名去重，因此可以反复执行。
    """
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(dst)
    try:
        rows = source.execute("SELECT * FROM users").fetchall()
        added = 0
        for row in rows:
            exists = target.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE",
                (row["username"],),
            ).fetchone()
            if exists:
                continue
            target.execute(
                "INSERT INTO users (id, username, password_hash, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    row["id"],
                    row["username"],
                    row["password_hash"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )
            added += 1
        target.commit()
        return added
    finally:
        target.close()
        source.close()


def _verify_key_pairing(config_db: Path, secret_key: Path) -> bool:
    """真去解一条供应商配置，确认库与密钥配对。

    这是本模块唯一能证明"搬迁成功"的检查：文件都在、行数也对，但如果密钥与
    加密的 ``api_key`` 不配对，配置照样用不了——而那时故障表现为
    ``SecretCipherError``，很容易被当成数据损坏。

    **调用前必须确认 ``secret_key`` 已存在**：``SecretCipher`` 在密钥文件缺失时
    会生成一把新的，于是这个"检查"会顺手写出一个文件，并对着它得出"解不开"的
    错误结论。
    """
    if not config_db.is_file() or not Path(secret_key).is_file():
        return False
    try:
        from finharness.config.crypto import SecretCipher
        from finharness.config.store import ConfigStore

        store = ConfigStore(config_db, cipher=SecretCipher(secret_key))
        for record in store.list_configs(user_id=""):
            # 解不出就会抛 SecretCipherError，正是我们要捕获的信号。
            store.resolve_key(record.id, user_id="")
            return True
        return True  # 配置库存在但没有条目
    except Exception:  # noqa: BLE001 - 校验失败由返回值表达，不向上抛
        return False


def _effective_secret_key(legacy: Path, configured: Path) -> Path | None:
    """搬迁之后实际生效的那把密钥。

    目标已有密钥时不覆盖（沿用目标那把）；否则将复制旧密钥，因此生效的是旧密钥。
    返回 None 表示两处都没有密钥——此时配置库不可能被解开。
    """
    if configured.is_file():
        return configured
    if legacy.is_file():
        return legacy
    return None


def migrate_legacy_state(settings, *, dry_run: bool = False) -> MigrationReport:
    """把旧布局的状态文件复制到当前配置的位置。

    * **复制而非移动**：原目录保留，出问题时原样可用。
    * **目标非空时合并账号**（而不是跳过）：跳过会让迁移后的依然登不上，
      守卫下次启动照旧失败，等于没搬。合并按用户名去重，可重复执行。
    * **密钥优先于配置库**：``config.db`` 里的 ``api_key`` 用主密钥加密，
      顺序颠倒会导致配置库解不开。
    * 目标已有密钥时**不覆盖**并单独报告：密钥换了会让已有配置全部失效，
      这是必须由人决定的事，不能由脚本猜。
    """
    legacy = legacy_state_paths(settings)
    configured = {key: _path_for(settings, key) for key in _LEGACY_FILES}

    state = detect_legacy_state(settings)
    if not state.legacy.has_any_data() and not legacy["secret_key"].is_file():
        raise MigrationError(
            f"在 {state.legacy.location} 没有找到可搬迁的状态数据，无需迁移。"
        )

    copied: list[str] = []
    skipped: list[str] = []
    unreadable_config = False
    merged_accounts = 0

    # 密钥优先：config.db 的解密依赖它。先确定密钥落在哪里（刚复制的沿用旧密钥，
    # 已存在的沿用目标密钥），后面的配置库判定都以它为基准。
    secret_src, secret_dst = legacy["secret_key"], configured["secret_key"]
    if secret_src.is_file() and secret_src.resolve() != secret_dst.resolve():
        if secret_dst.is_file():
            # 两处都有密钥：不猜，交由人决定（可能加密的是不同的数据集）。
            skipped.append("secret_key")
        else:
            if not dry_run:
                _copy_secret(secret_src, secret_dst)
            copied.append("secret_key")

    for key in ("auth_db", "memory_db"):
        src, dst = legacy[key], configured[key]
        if not src.is_file() or src.resolve() == dst.resolve():
            continue
        if _destination_is_populated(dst, key):
            if key == "auth_db" and not dry_run:
                # 目标已有账号：补进缺失的那些，让旧账号重新可登录。
                merged_accounts = _merge_users(src, dst)
                copied.append(f"{key}(合并 {merged_accounts} 个账号)")
            else:
                skipped.append(key)
            continue
        if not dry_run:
            _copy_sqlite(src, dst)
        copied.append(key)

    # 配置库单独处理：它的 api_key 是被主密钥加密的，因此只有当**搬迁后实际
    # 生效的那把密钥**确实能解开它时才允许复制。否则复制过去的是一个读不出来的
    # 库——"搬过去了但用不了"，正是本模块要防的故障形态。
    config_src, config_dst = legacy["config_db"], configured["config_db"]
    effective_secret = _effective_secret_key(secret_src, secret_dst)
    if config_src.is_file() and config_src.resolve() != config_dst.resolve():
        if _destination_is_populated(config_dst, "config_db"):
            skipped.append("config_db")
        elif effective_secret is None or not _verify_key_pairing(config_src, effective_secret):
            skipped.append("config_db")
            unreadable_config = True
        else:
            if not dry_run:
                _copy_sqlite(config_src, config_dst)
            copied.append("config_db")

    verified = True
    if not dry_run:
        verified = _verify_key_pairing(configured["config_db"], configured["secret_key"])

    return MigrationReport(
        copied=tuple(copied),
        skipped=tuple(skipped),
        verified=verified,
        dry_run=dry_run,
        merged_accounts=merged_accounts,
        unreadable_config=unreadable_config,
    )


# -- 命令行入口 ----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """``python -m finharness.config.state_migration --settings settings.json``"""
    from finharness.config.settings import Settings, SettingsError

    parser = argparse.ArgumentParser(
        description="把旧布局（data_cache/）的状态数据搬迁到 paths.state_dir。"
    )
    parser.add_argument("--settings", default=None, help="settings.json 路径")
    parser.add_argument(
        "--apply", action="store_true", help="真正执行（默认只预览，不写任何文件）"
    )
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_file(args.settings)
    except SettingsError as exc:
        print(f"配置加载失败：{exc}")
        return 2

    state = detect_legacy_state(settings)
    print(f"旧位置：{state.legacy.location} -> {state.legacy.describe()}")
    print(f"新位置：{state.configured.location} -> {state.configured.describe()}")

    try:
        report = migrate_legacy_state(settings, dry_run=not args.apply)
    except MigrationError as exc:
        print(f"\n无需迁移：{exc}")
        return 1

    print()
    print(report.describe())
    if report.dry_run:
        print("\n以上为预览。确认后加 --apply 执行。")
    else:
        print(f"\n原目录 {state.legacy.location} 已保留作备份，确认无误后可自行删除。")
    return 0


if __name__ == "__main__":  # pragma: no cover - 入口
    raise SystemExit(main())
