#!/usr/bin/env python3
"""Warn on tight coupling between Esper subsystems.

Scans modules under ``src/esper`` and emits a warning when a package imports
another subsystem directly instead of going through Leyline contracts or
approved integration surfaces.

The check exits with status 0 by default so it can run in CI without failing
pipelines. Pass ``--strict`` to make warnings fatal when you want to ratchet
standards locally.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Iterable

# Root paths
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "esper"

# Allowlisted targets that represent shared infrastructure rather than tight
# coupling between subsystems.
ALLOWED_TARGETS = {
    "leyline",
    "core",
    "security",
    "oona",
    "weatherlight",
}

# Subsystems that are explicitly allowed to import each other per ADR-001.
TIGHTLY_COUPLED = {
    frozenset({"tamiyo", "tolaria"}),
    frozenset({"tamiyo", "kasmina"}),
    frozenset({"tolaria", "kasmina"}),
}


@dataclass(frozen=True)
class ImportWarning:
    source_package: str
    target_package: str
    module: str
    filename: Path
    lineno: int

    def format(self) -> str:
        rel = self.filename.relative_to(REPO_ROOT)
        return (
            f"WARNING: {rel}:{self.lineno} imports esper.{self.target_package} "
            f"({self.source_package} -> {self.target_package})"
        )


def discover_modules() -> Iterable[Path]:
    for path in SRC_ROOT.rglob("*.py"):
        # Ignore __pycache__ or generated files
        if "__pycache__" in path.parts:
            continue
        yield path


def package_of(path: Path) -> str | None:
    try:
        rel = path.relative_to(SRC_ROOT)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    return parts[0]


def top_level_from_import(name: str) -> str | None:
    if not name.startswith("esper"):
        return None
    parts = name.split(".")
    if len(parts) < 2:
        return None
    return parts[1]


def check_file(path: Path) -> list[ImportWarning]:
    source_package = package_of(path)
    if source_package is None:
        return []

    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    warnings: list[ImportWarning] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = top_level_from_import(alias.name)
                if target is None:
                    continue
                if _is_violation(source_package, target):
                    warnings.append(
                        ImportWarning(
                            source_package,
                            target,
                            alias.name,
                            path,
                            getattr(node, "lineno", 0),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative import stays within the same package
                continue
            if not node.module:
                continue
            target = top_level_from_import(node.module)
            if target is None:
                continue
            if _is_violation(source_package, target):
                warnings.append(
                    ImportWarning(
                        source_package,
                        target,
                        node.module,
                        path,
                        getattr(node, "lineno", 0),
                    )
                )
    return warnings


def _is_violation(source: str, target: str) -> bool:
    if source == target:
        return False
    if target in ALLOWED_TARGETS:
        return False
    if frozenset({source, target}) in TIGHTLY_COUPLED:
        return False
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Warn on cross-subsystem imports outside Leyline."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any warnings are found.",
    )
    args = parser.parse_args(argv)

    all_warnings: list[ImportWarning] = []
    for module_path in discover_modules():
        all_warnings.extend(check_file(module_path))

    for warning in sorted(all_warnings, key=lambda w: (w.filename, w.lineno)):
        print(warning.format(), file=sys.stderr)

    if args.strict and all_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
