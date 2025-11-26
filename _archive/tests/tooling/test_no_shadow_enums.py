from __future__ import annotations

import subprocess
from pathlib import Path


def test_no_shadow_enums_script_runs_ok() -> None:
    script = Path("scripts/check_shared_types.py").resolve()
    assert script.exists()
    # Run the checker; assert it returns 0 on the current repo
    proc = subprocess.run(
        ["python3", str(script), "--root", "src/esper"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    combined = (proc.stdout + proc.stderr).lower()
    assert "ok" in combined or "no shadow enums" in combined
