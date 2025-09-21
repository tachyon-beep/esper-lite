#!/usr/bin/env python3
"""Guardrail to ensure shared enums remain centralised in Leyline."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "src" / "esper"

ALLOWED_ENUM_PATHS = {
    PY_SRC / "leyline" / "_generated" / "leyline_pb2.py",
    PY_SRC / "leyline" / "_generated" / "leyline_pb2.pyi",
}

pattern = re.compile(r"class\s+\w+\([^)]*Enum[^)]*\):")
violations: list[tuple[Path, str]] = []

for path in PY_SRC.rglob("*.py"):
    if any(str(path).startswith(str(allowed)) for allowed in ALLOWED_ENUM_PATHS):
        continue
    text = path.read_text(encoding="utf-8")
    for match in pattern.finditer(text):
        violations.append((path.relative_to(ROOT), match.group(0).strip()))

if violations:
    print("Detected non-canonical enum definitions; move enums into Leyline:")
    for path, snippet in violations:
        print(f" - {path}: {snippet}")
    sys.exit(1)

def main() -> int:
    return 0


if __name__ == "__main__":
    sys.exit(main())
