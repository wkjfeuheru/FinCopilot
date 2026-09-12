"""Runtime compatibility shims required before pandas/akshare are imported.

pandas 3.0 can back its string dtype with pyarrow. pyarrow's RE2 regex engine
does not implement ``\\uXXXX`` escapes, which akshare relies on internally, so
string operations raise ``ArrowInvalid: invalid escape sequence``. Pinning the
string storage back to the Python backend keeps akshare working while parquet
reads and writes still go through pyarrow.
"""

from __future__ import annotations

_applied = False


def apply_data_runtime_compat() -> None:
    """Idempotently restore the Python string backend for pandas."""
    global _applied
    if _applied:
        return
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        return
    pd.options.mode.string_storage = "python"
    _applied = True
