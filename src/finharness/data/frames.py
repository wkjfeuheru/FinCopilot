"""持久化工具派生的数据框，使其可被引用并重新绘图。

``make_chart`` 通过引用的 ``parquet_path`` 复用数据：一个已经抓取过
某数据框的会话，不应仅为绘图而重新抓取。抓取所得的数据框从数据缓存
获得该路径，但*派生*数据框（回测的净值序列、计算得到的分解结果）
由工具生成而非抓取，因此没有任何地方将其写入磁盘，其引用也就
没有可供复用的路径。

本模块是弥合这一缺口的地方：工具把它的数据框交给这里，取回一个绝对
parquet 路径，附加到自己的 ``RawData`` 上。文件存放在数据缓存之下，
而该目录本就是 ``read_file`` 的读取路径，因此数据框既可复用
又可查看。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def persist_frame(df: pd.DataFrame, *, cache_dir: str | Path, name: str) -> str | None:
    """将 ``df`` 写入 ``<cache_dir>/frames/<name>_<digest>.parquet``。

    返回绝对路径；当数据框为空或无法写入时返回 ``None``。派生数据框的
    缓存只是一种优化，绝非正确性要求：此处失败不得导致生成该数据框的
    工具失败，因此调用方保留其内存中的数据框，仅失去复用而已。
    """
    if df is None or not len(df):
        return None
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))[:40] or "frame"
    try:
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()[:12]
        directory = Path(cache_dir) / "frames"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{safe}_{digest}.parquet"
        if not path.is_file():
            df.to_parquet(path, engine="pyarrow", index=False)
        return str(path.resolve())
    except (OSError, ValueError, ImportError, TypeError):
        return None
