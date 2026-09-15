"""M5 demo 脚本必须保持可运行。

demo 本身需要实时服务器和真实 provider，因此无法在离线测试集中运行。
能以低成本检查的是：脚本仍可启动并解析其参数——否则，一个损坏的 import
或一个拼错的 flag 只会在有人手动运行 demo 时才暴露出来。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "demo.py"


def test_demo_script_help_runs():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--demo" in completed.stdout
    assert "--base-url" in completed.stdout
