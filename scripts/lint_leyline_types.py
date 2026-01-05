#!/usr/bin/env python3
"""Enforce leyline type boundaries.

All shared types (enums, dataclasses, protocols, typeddicts, namedtuples)
must be in leyline/ unless allowed by leyline_boundaries.yaml.

Usage:
    python scripts/lint_leyline_types.py [--verbose]

Exit codes:
    0 - All checks passed
    1 - Violations found or configuration error
"""
from __future__ import annotations

import argparse
import ast
import sys
from datetime import date
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import yaml

KNOWN_TYPE_KINDS = {"enum", "dataclass", "protocol", "typeddict", "namedtuple"}


class TypeDef(NamedTuple):
    """A type definition found in the codebase."""

    path: str  # relative path from project root
    kind: str  # enum | dataclass | protocol | typeddict | namedtuple
    name: str  # class name
    line: int  # line number for error reporting


class Whitelist:
    """Manages the leyline boundary whitelist."""

    def __init__(self, config_path: Path) -> None:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("leyline_boundaries.yaml must be a mapping at top level")

        self.allow_paths: list[str] = []
        allow_paths = data.get("allow_paths", [])
        if not isinstance(allow_paths, list):
            raise ValueError("allow_paths must be a list")
        for entry in allow_paths:
            if not isinstance(entry, dict):
                raise ValueError("allow_paths entries must be mappings")
            path = entry.get("path")
            reason = entry.get("reason")
            if not isinstance(path, str) or not path:
                raise ValueError("allow_paths entries must have non-empty 'path'")
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"allow_paths '{path}' must include non-empty 'reason'")
            self.allow_paths.append(path)

        self.allow_hits: dict[str, dict] = {}
        allow_hits = data.get("allow_hits", [])
        if not isinstance(allow_hits, list):
            raise ValueError("allow_hits must be a list")
        for entry in allow_hits:
            if not isinstance(entry, dict):
                raise ValueError("allow_hits entries must be mappings")
            key = entry.get("key")
            owner = entry.get("owner")
            reason = entry.get("reason")
            expires_raw = entry.get("expires")

            if not isinstance(key, str) or not key:
                raise ValueError("allow_hits entries must have non-empty 'key'")
            if not isinstance(owner, str) or not owner:
                raise ValueError(f"allow_hits '{key}' must include non-empty 'owner'")
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"allow_hits '{key}' must include non-empty 'reason'")

            parts = key.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid allow_hits key format (expected path:type_kind:ClassName): {key}"
                )
            kind = parts[1]
            if kind not in KNOWN_TYPE_KINDS:
                raise ValueError(f"Unknown type_kind '{kind}' in allow_hits key: {key}")

            if expires_raw is not None:
                if not isinstance(expires_raw, str) or not expires_raw:
                    raise ValueError(f"allow_hits '{key}' has invalid expires value")
                try:
                    date.fromisoformat(expires_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"allow_hits '{key}' has invalid expires date: {expires_raw}"
                    ) from exc

            if key in self.allow_hits:
                raise ValueError(f"Duplicate allow_hits key: {key}")
            self.allow_hits[key] = entry

        self.today = date.today()

    def is_allowed(self, typedef: TypeDef) -> tuple[bool, str | None]:
        """Check if typedef is allowed.

        Returns:
            (allowed, warning_message_if_expired)
        """
        # Check broad path allowances (glob patterns)
        for pattern in self.allow_paths:
            if fnmatch(typedef.path, pattern):
                return True, None

        # Check narrow allowances (exact key match)
        key = f"{typedef.path}:{typedef.kind}:{typedef.name}"
        if key in self.allow_hits:
            hit = self.allow_hits[key]
            if "expires" in hit:
                expires = date.fromisoformat(hit["expires"])
                if expires < self.today:
                    return False, f"EXPIRED: {key} (was {hit['expires']})"
            return True, None

        return False, None


def get_base_name(node: ast.expr) -> str | None:
    """Extract the base class name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        # Handle Generic[T] style
        return get_base_name(node.value)
    return None


def classify_class(node: ast.ClassDef) -> str | None:
    """Determine if class is enum/dataclass/protocol/typeddict/namedtuple."""
    # Check decorators for @dataclass
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "dataclass":
            return "dataclass"
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            if dec.func.id == "dataclass":
                return "dataclass"
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            if dec.func.attr == "dataclass":
                return "dataclass"

    # Check base classes
    for base in node.bases:
        base_name = get_base_name(base)
        if base_name in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"):
            return "enum"
        if base_name == "Protocol":
            return "protocol"
        if base_name == "TypedDict":
            return "typeddict"
        if base_name == "NamedTuple":
            return "namedtuple"

    return None


def find_type_definitions(path: Path, tree: ast.AST) -> list[TypeDef]:
    """Find all enum/dataclass/protocol/typeddict/namedtuple definitions."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            kind = classify_class(node)
            if kind:
                results.append(TypeDef(str(path), kind, node.name, node.lineno))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce leyline type boundaries"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all checked files"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("leyline_boundaries.yaml"),
        help="Path to whitelist config",
    )
    args = parser.parse_args()

    config_path: Path = args.config
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return 1

    try:
        whitelist = Whitelist(config_path)
    except ValueError as exc:
        print(f"ERROR: Invalid whitelist config: {exc}")
        return 1
    violations: list[str] = []
    warnings: list[str] = []
    checked_files = 0
    checked_types = 0
    used_allow_hit_keys: set[str] = set()

    for path in sorted(Path("src/esper").rglob("*.py")):
        # Skip __pycache__ directories
        if "__pycache__" in path.parts:
            continue

        # leyline is always allowed - that's the whole point
        if "leyline" in path.parts:
            continue

        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:
            print(f"WARNING: Could not parse {path}: {e}")
            continue

        checked_files += 1
        typedefs = find_type_definitions(path, tree)

        for typedef in typedefs:
            checked_types += 1
            key = f"{typedef.path}:{typedef.kind}:{typedef.name}"
            if key in whitelist.allow_hits:
                used_allow_hit_keys.add(key)
            allowed, warning = whitelist.is_allowed(typedef)

            if warning:
                warnings.append(warning)

            if not allowed:
                violations.append(
                    f"{typedef.path}:{typedef.line}: "
                    f"{typedef.kind} '{typedef.name}' not in leyline or whitelist"
                )

            if args.verbose and allowed:
                print(f"OK: {typedef.path}:{typedef.line} {typedef.kind} {typedef.name}")

    # Print summary
    print(f"\nChecked {checked_files} files, {checked_types} type definitions")
    stale_allow_hits = sorted(set(whitelist.allow_hits.keys()) - used_allow_hit_keys)
    print(f"Stale whitelist entries: {len(stale_allow_hits)}")

    for w in warnings:
        print(f"WARNING: {w}")

    if stale_allow_hits:
        print("\nSTALE WHITELIST ENTRIES FOUND:\n")
        for key in stale_allow_hits:
            print(f"  {key}")
        print("\nTo fix: remove or update these keys in leyline_boundaries.yaml")
        return 1

    if violations:
        print(f"\n{len(violations)} violation(s) found:\n")
        for v in violations:
            print(f"  ERROR: {v}")
        print(
            "\nTo fix: either move the type to leyline/, or add it to leyline_boundaries.yaml"
        )
        return 1

    print("All type definitions are in leyline or whitelisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
