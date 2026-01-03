#!/usr/bin/env python3
"""Detect unauthorized defensive programming patterns.

This linter finds patterns that can hide bugs:
  - getattr() - can mask missing attributes (STRICT: always flagged)
  - hasattr() - can mask integration errors (STRICT: always flagged)
  - Silent exception handling - can swallow real errors (STRICT: always flagged)
  - isinstance() - can mask wrong types (AUDIT: flagged with --audit)
  - .get() - can mask missing dict keys (AUDIT: flagged with --audit)

Strict patterns require explicit whitelisting. Audit patterns are allowed
by default but can be reviewed with --audit for code quality.

Usage:
    python scripts/lint_defensive_patterns.py [--verbose]
    python scripts/lint_defensive_patterns.py --audit  # Include isinstance/.get()
    python scripts/lint_defensive_patterns.py --generate-keys  # Output keys for whitelisting

Exit codes:
    0 - All checks passed
    1 - Violations found or configuration error
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from datetime import date
from fnmatch import fnmatch
from pathlib import Path

import yaml

# Patterns that are always flagged (high risk of hiding bugs)
STRICT_PATTERNS = {"getattr", "hasattr", "silent_except", "bare_except"}

# Patterns only flagged in audit mode (legitimate uses are common)
AUDIT_PATTERNS = {"isinstance", "get"}


@dataclass
class PatternHit:
    """A defensive pattern found in the codebase."""

    path: str  # relative path from project root
    pattern: str  # isinstance | get | getattr | hasattr | silent_except | bare_except
    function: str  # containing function/method name, or <module> for top-level
    line: int  # line number
    code_snippet: str  # the actual code for context
    occurrence: int = 1  # 1-based index if multiple in same function

    @property
    def key(self) -> str:
        """Generate whitelist key for this hit."""
        if self.occurrence > 1:
            return f"{self.path}:{self.function}:{self.pattern}:{self.occurrence}"
        return f"{self.path}:{self.function}:{self.pattern}"


class PatternVisitor(ast.NodeVisitor):
    """AST visitor that finds defensive programming patterns."""

    def __init__(self, path: str, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.hits: list[PatternHit] = []
        self._function_stack: list[str] = []
        # Track occurrences per (function, pattern) for numbering
        self._occurrences: dict[tuple[str, str], int] = {}

    def _current_function(self) -> str:
        return self._function_stack[-1] if self._function_stack else "<module>"

    def _get_code_snippet(self, node: ast.AST) -> str:
        """Extract the source code for a node."""
        if hasattr(node, "lineno"):
            line_idx = node.lineno - 1
            if 0 <= line_idx < len(self.source_lines):
                return self.source_lines[line_idx].strip()
        return "<unavailable>"

    def _record_hit(self, node: ast.AST, pattern: str) -> None:
        """Record a pattern hit with proper occurrence numbering."""
        func = self._current_function()
        key = (func, pattern)
        self._occurrences[key] = self._occurrences.get(key, 0) + 1
        occurrence = self._occurrences[key]

        self.hits.append(
            PatternHit(
                path=self.path,
                pattern=pattern,
                function=func,
                line=getattr(node, "lineno", 0),
                code_snippet=self._get_code_snippet(node),
                occurrence=occurrence,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        # Check for isinstance(), getattr(), hasattr()
        if isinstance(node.func, ast.Name):
            if node.func.id == "isinstance":
                self._record_hit(node, "isinstance")
            elif node.func.id == "getattr":
                self._record_hit(node, "getattr")
            elif node.func.id == "hasattr":
                self._record_hit(node, "hasattr")

        # Check for .get() method calls
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "get":
                self._record_hit(node, "get")

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Bare except (catches everything including SystemExit, KeyboardInterrupt)
        if node.type is None:
            self._record_hit(node, "bare_except")
        # Check for silent exception handling
        elif self._is_silent_handler(node):
            self._record_hit(node, "silent_except")

        self.generic_visit(node)

    def _is_silent_handler(self, node: ast.ExceptHandler) -> bool:
        """Check if exception handler silently swallows errors."""
        if not node.body:
            return True

        # Single-statement handlers that do nothing useful
        if len(node.body) == 1:
            stmt = node.body[0]
            # pass statement
            if isinstance(stmt, ast.Pass):
                return True
            # continue statement (in loops)
            if isinstance(stmt, ast.Continue):
                return True
            # Bare raise is OK (re-raises)
            if isinstance(stmt, ast.Raise) and stmt.exc is None:
                return False
            # ... (ellipsis) sometimes used as pass equivalent
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value is ...:
                    return True

        return False


class Whitelist:
    """Manages the defensive patterns whitelist."""

    def __init__(self, config_path: Path) -> None:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        self.version = data.get("version", 1)
        self.always_prohibited: set[str] = {
            p["pattern"] for p in data.get("always_prohibited", [])
        }
        self.allow_paths: list[str] = [
            p["path"] for p in data.get("allow_paths", [])
        ]
        self.allow_hits: dict[str, dict] = {
            h["key"]: h for h in data.get("allow_hits", [])
        }
        self.today = date.today()

    def is_allowed(self, hit: PatternHit) -> tuple[bool, str | None]:
        """Check if pattern hit is allowed.

        Returns:
            (allowed, warning_or_error_message)
        """
        # Always-prohibited patterns cannot be whitelisted
        if hit.pattern in self.always_prohibited:
            return False, f"PROHIBITED: {hit.pattern} can never be whitelisted"

        # Check broad path allowances (glob patterns)
        for pattern in self.allow_paths:
            if fnmatch(hit.path, pattern):
                return True, None

        # Check narrow allowances (exact key match)
        # Try with occurrence first, then without
        keys_to_try = [hit.key]
        if hit.occurrence > 1:
            # Also try base key without occurrence for blanket function allowance
            base_key = f"{hit.path}:{hit.function}:{hit.pattern}"
            keys_to_try.append(base_key)

        for key in keys_to_try:
            if key in self.allow_hits:
                entry = self.allow_hits[key]
                if "expires" in entry:
                    expires = date.fromisoformat(entry["expires"])
                    if expires < self.today:
                        return False, f"EXPIRED: {key} (was {entry['expires']})"
                return True, None

        return False, None


def find_patterns(path: Path) -> list[PatternHit]:
    """Find all defensive programming patterns in a file."""
    try:
        source = path.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"WARNING: Could not parse {path}: {e}", file=sys.stderr)
        return []

    source_lines = source.splitlines()
    visitor = PatternVisitor(str(path), source_lines)
    visitor.visit(tree)
    return visitor.hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect unauthorized defensive programming patterns"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all checked files"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Also flag isinstance() and .get() (normally allowed)",
    )
    parser.add_argument(
        "--generate-keys",
        action="store_true",
        help="Generate whitelist keys for all hits (for adding to YAML)",
    )
    parser.add_argument(
        "--strict-only",
        action="store_true",
        help="Only show strict patterns in --generate-keys output",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("defensive_patterns.yaml"),
        help="Path to whitelist config",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("src/esper"),
        help="Path to scan (default: src/esper)",
    )
    args = parser.parse_args()

    # Determine which patterns to flag
    active_patterns = set(STRICT_PATTERNS)
    if args.audit:
        active_patterns |= AUDIT_PATTERNS

    config_path: Path = args.config
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return 1

    whitelist = Whitelist(config_path)
    violations: list[tuple[PatternHit, str | None]] = []
    warnings: list[str] = []
    checked_files = 0
    total_hits = 0
    allowed_hits = 0

    scan_path: Path = args.path
    for path in sorted(scan_path.rglob("*.py")):
        # Skip __pycache__ directories
        if "__pycache__" in path.parts:
            continue

        checked_files += 1
        hits = find_patterns(path)

        for hit in hits:
            total_hits += 1

            # Skip audit patterns unless --audit is specified
            if hit.pattern in AUDIT_PATTERNS and not args.audit:
                continue

            # Skip non-strict patterns in --strict-only mode
            if args.strict_only and hit.pattern not in STRICT_PATTERNS:
                continue

            allowed, message = whitelist.is_allowed(hit)

            if message and "EXPIRED" in message:
                warnings.append(message)

            if args.generate_keys:
                # Output mode: show all hits as potential whitelist entries
                tier = "STRICT" if hit.pattern in STRICT_PATTERNS else "AUDIT"
                print(f"  - key: \"{hit.key}\"")
                print(f"    owner: \"\"")
                print(f"    reason: \"\"  # [{tier}] {hit.code_snippet[:50]}")
                print()
            elif not allowed:
                violations.append((hit, message))
            else:
                allowed_hits += 1
                if args.verbose:
                    print(f"OK: {hit.key} ({hit.code_snippet[:50]}...)")

    if args.generate_keys:
        mode = "strict only" if args.strict_only else ("audit" if args.audit else "strict")
        print(f"\n# Found {total_hits} total hits in {checked_files} files (showing: {mode})")
        return 0

    # Print summary
    mode = "audit mode (strict + isinstance/get)" if args.audit else "strict mode"
    print(f"\nChecked {checked_files} files in {mode}")
    print(f"  Total patterns found: {total_hits}")
    print(f"  Patterns checked: {allowed_hits + len(violations)}")
    print(f"  Allowed: {allowed_hits}")
    print(f"  Violations: {len(violations)}")

    for w in warnings:
        print(f"WARNING: {w}")

    if violations:
        print(f"\n{len(violations)} VIOLATION(S) FOUND:\n")
        for hit, message in violations:
            print(f"  {hit.path}:{hit.line}: {hit.pattern} in {hit.function}()")
            print(f"    Code: {hit.code_snippet[:80]}")
            print(f"    Key:  {hit.key}")
            if message:
                print(f"    Note: {message}")
            print()

        print("To fix:")
        print("  1. PREFERRED: Remove the defensive pattern and fix the underlying issue")
        print("  2. If legitimate: Add the key to defensive_patterns.yaml with justification")
        print("\nRun with --generate-keys to output whitelist entries for all hits")
        return 1

    print("\nAll defensive patterns are whitelisted or absent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
