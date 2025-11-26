#!/usr/bin/env python3
"""Guardrail: prevent local Enum definitions that shadow Leyline contracts.

Scans `src/esper` for Python Enums and fails if any are found outside the
allowed Leyline-generated subtree: `src/esper/leyline/_generated`.

Usage:
  python3 scripts/check_shared_types.py [--root src/esper]
Exits 0 if no violations; exits 1 and prints details if violations are found.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable


ENUM_BASE_NAMES = {"Enum", "IntEnum", "Flag", "IntFlag"}


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        # Skip Leyline-generated protobufs
        try:
            rel = path.as_posix()
        except Exception:
            rel = str(path)
        if "/leyline/_generated/" in rel or rel.endswith("/leyline/_generated/__init__.py"):
            continue
        yield path


def find_shadow_enums(file_path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line, class_name, base_repr) for enum violations in file."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    enum_module_aliases: set[str] = set()
    enum_base_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "enum":
                    enum_module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "enum":
                for alias in node.names:
                    base = alias.name
                    if base in ENUM_BASE_NAMES:
                        enum_base_aliases.add(alias.asname or base)

    violations: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            # Case: Name (Enum alias)
            if isinstance(base, ast.Name):
                name = base.id
                if name in ENUM_BASE_NAMES or name in enum_base_aliases:
                    violations.append((node.lineno, node.name, name))
                    break
            # Case: Attribute (e.g., enum.Enum or alias.Enum)
            elif isinstance(base, ast.Attribute):
                # attr: Enum/IntEnum/...; value: Name of module alias
                try:
                    attr = base.attr
                    val = base.value.id if isinstance(base.value, ast.Name) else None
                except Exception:
                    attr, val = None, None
                if attr in ENUM_BASE_NAMES and val in (enum_module_aliases or {"enum"}):
                    violations.append((node.lineno, node.name, f"{val}.{attr}"))
                    break

    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="src/esper", help="Root directory to scan")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    all_violations: list[tuple[Path, int, str, str]] = []
    for file in iter_python_files(root):
        violations = find_shadow_enums(file)
        for line, cls, base in violations:
            all_violations.append((file, line, cls, base))

    if not all_violations:
        print("[check_shared_types] OK: no shadow Enums found.")
        return 0

    print("[check_shared_types] ERROR: Found local Enum definitions (disallowed):")
    for file, line, cls, base in sorted(all_violations):
        try:
            rel = file.relative_to(Path.cwd())
        except Exception:
            rel = file
        print(f"  - {rel}:{line}: class {cls}({base})")
    return 1


if __name__ == "__main__":
    sys.exit(main())

