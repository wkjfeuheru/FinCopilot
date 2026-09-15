"""在导入 pandas/akshare 之前所需的运行时兼容垫片。

pandas 3.0 可以用 pyarrow 作为其字符串 dtype 的后端。pyarrow 的 RE2 正则
引擎未实现 ``\\uXXXX`` 转义，而 akshare 内部依赖它，于是字符串操作会抛出
``ArrowInvalid: invalid escape sequence``。把字符串存储固定回 Python 后端，
可在 parquet 读写仍经由 pyarrow 的同时保持 akshare 正常工作。
"""

from __future__ import annotations

_applied = False


def apply_data_runtime_compat() -> None:
    """幂等地把 pandas 的字符串后端恢复为 Python。"""
    global _applied
    if _applied:
        return
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas 是硬依赖
        return
    pd.options.mode.string_storage = "python"
    _applied = True
