#!/usr/bin/env python3
"""CLI entrypoint for the RC1 cross-system performance harness."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from esper.tools.rc1_harness import main


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    path = main()
    print(path)
