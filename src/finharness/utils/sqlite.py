"""共享的 SQLite 短连接基础设施。

多个 store（users / config / memory / usage / trace / compute queue / data cache）
各自重写过一遍 ``sqlite3.connect(timeout=10.0)`` + ``row_factory = Row`` + ``ping``
样板。集中到这里后，"短连接 + 行工厂"的语义只有一处定义，差异（是否允许跨线程、
是否开启外键、超时秒数）通过构造参数表达。

注意这是短连接模型：每次操作 ``with self._connect() as connection`` 打开一条连接，
with 块退出时提交/回滚但**不关闭**连接（由 GC 回收），与重构前完全一致。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteStore:
    """短连接 + WAL 的 SQLite 存储基类。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        check_same_thread: bool = True,
        foreign_keys: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_same_thread = check_same_thread
        self._foreign_keys = foreign_keys
        self._timeout = timeout

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=self._timeout,
            check_same_thread=self._check_same_thread,
        )
        connection.row_factory = sqlite3.Row
        # 与 connect(timeout=) 同源，但显式落到连接上：busy 等待不依赖驱动的隐式映射。
        connection.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")
        if self._foreign_keys:
            connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ping(self) -> None:
        """就绪检查：库可连接且可查询。"""
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
