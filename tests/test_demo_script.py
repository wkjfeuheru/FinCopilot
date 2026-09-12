"""The M5 demo script must stay runnable.

The demo itself needs a live server and a real provider, so it cannot run in the
offline suite. What *can* be checked cheaply is that the script still starts and
parses its arguments — a broken import or a typo'd flag would otherwise only
surface when someone tries to run the demo by hand.
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
