"""受限计算子进程入口；仅从服务端注册的函数路径加载 handler。"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

from finharness.compute.protocol import extract_task_package


def main() -> None:
    archive, handler_name, kind = sys.argv[1:4]
    module_name, qualname = handler_name.split(":", 1)
    handler = importlib.import_module(module_name)
    for part in qualname.split("."):
        handler = getattr(handler, part)
    with tempfile.TemporaryDirectory(prefix="finh-compute-child-") as temporary:
        input_dir = Path(temporary) / "input"
        extract_task_package(archive, input_dir)
        result = handler({"kind": kind}, input_dir)
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
