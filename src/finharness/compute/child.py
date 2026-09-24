"""受限计算子进程入口；仅从服务端注册的函数路径加载 handler。"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

from finharness.compute.protocol import extract_task_package


def main() -> None:
    # 父进程先绑定 Windows Job Object，随后才允许 handler 生成后代。
    if sys.stdin.buffer.read(1) != b"1":
        raise SystemExit(1)
    archive, handler_name, kind, scratch = sys.argv[1:5]
    module_name, qualname = handler_name.split(":", 1)
    handler = importlib.import_module(module_name)
    for part in qualname.split("."):
        handler = getattr(handler, part)
    # 输入必须解到父进程创建的清理目录内：解包与清理由同一属主完成，
    # 父进程的 rmtree 收尾才总是成立（Linux 降权后子进程建不了目录）。
    input_dir = Path(scratch) / "input"
    extract_task_package(archive, input_dir)
    try:
        result = handler({"kind": kind}, input_dir)
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
