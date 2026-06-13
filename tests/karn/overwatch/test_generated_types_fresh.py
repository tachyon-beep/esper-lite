"""UI-003 freshness guard: committed sanctum.ts must match the schema generator.

The Overwatch TypeScript types in ``src/esper/karn/overwatch/web/src/types/sanctum.ts``
are mechanically generated from the Python Sanctum schema by
``scripts/generate_overwatch_types.py``. Nothing else fails when the committed
file drifts from the schema, so this test regenerates the types from the current
Python schema and asserts the committed file is byte-identical.

If this test fails, the schema changed without regenerating the TypeScript. Fix
it by running, from the repo root::

    uv run python scripts/generate_overwatch_types.py > \
        src/esper/karn/overwatch/web/src/types/sanctum.ts
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "scripts" / "generate_overwatch_types.py"
COMMITTED_TS = (
    REPO_ROOT
    / "src"
    / "esper"
    / "karn"
    / "overwatch"
    / "web"
    / "src"
    / "types"
    / "sanctum.ts"
)

REGEN_COMMAND = (
    "uv run python scripts/generate_overwatch_types.py > "
    "src/esper/karn/overwatch/web/src/types/sanctum.ts"
)


def _generate_types() -> str:
    """Run the generator and capture its stdout (the TypeScript source)."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_committed_sanctum_ts_is_fresh() -> None:
    """The committed sanctum.ts must equal the generator's current output."""
    expected = _generate_types()
    actual = COMMITTED_TS.read_text()

    assert actual == expected, (
        "Committed Overwatch TypeScript types are stale relative to the Python "
        "Sanctum schema. The generated sanctum.ts no longer matches "
        "scripts/generate_overwatch_types.py output.\n\n"
        f"Regenerate it from the repo root with:\n    {REGEN_COMMAND}\n"
    )
