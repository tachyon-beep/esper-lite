"""WP0 acceptance: ensure no local Enums under src/esper.

This test integrates the repo's guard script into CI by invoking it and
asserting it exits successfully (no shadow enums outside Leyline protobufs).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_shared_enum_guard_passes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "check_shared_types.py"
    assert script.exists(), f"Missing guard script: {script}"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        raise AssertionError(
            f"Enum guard failed with code {result.returncode}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
