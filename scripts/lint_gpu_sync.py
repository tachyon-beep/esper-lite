#!/usr/bin/env python3
"""Enforce a whitelist for GPU synchronization points.

PyTorch training throughput collapses when we introduce accidental CPU↔GPU
synchronization in hot paths (e.g., implicit sync via Tensor.item()).

This linter detects *potential* GPU sync points:
  - Tensor.item(), Tensor.tolist(), Tensor.cpu(), Tensor.numpy()
  - torch.cuda.synchronize(...)
  - stream/event.synchronize()

All hits must be whitelisted in gpu_sync_whitelist.yaml (or via allow_paths).

Usage:
    python scripts/lint_gpu_sync.py
    python scripts/lint_gpu_sync.py --generate-keys
    python scripts/lint_gpu_sync.py --generate-baseline

Exit codes:
    0 - All checks passed (or key generation mode)
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


SYNC_METHOD_PATTERNS = {
    "item",
    "tolist",
    "cpu",
    "numpy",
    "synchronize",  # stream/event synchronize()
    "to_cpu",  # tensor.to("cpu") / to(device="cpu")
    "cuda_synchronize",  # torch.cuda.synchronize(...)
}


@dataclass
class SyncHit:
    """A potential GPU synchronization point found in the codebase."""

    path: str  # relative path from project root
    pattern: str  # item | tolist | cpu | numpy | synchronize | to_cpu | cuda_synchronize
    scope: str  # qualname-like scope (<module>, func, Class.method, etc)
    line: int  # 1-based line number
    code_snippet: str  # source line for context
    occurrence: int = 1  # 1-based index if multiple in same scope

    @property
    def key(self) -> str:
        """Generate whitelist key for this hit."""
        if self.occurrence > 1:
            return f"{self.path}:{self.scope}:{self.pattern}:{self.occurrence}"
        return f"{self.path}:{self.scope}:{self.pattern}"


def _dotted_name(node: ast.AST) -> str | None:
    """Return dotted name for a Name/Attribute chain, else None.

    Examples:
      - torch.cuda.synchronize -> "torch.cuda.synchronize"
      - env_state.stream.synchronize -> "env_state.stream.synchronize"
    """
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _is_cpu_device_expr(node: ast.AST) -> bool:
    """Return True if node is a simple literal/device expression for CPU."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value == "cpu"
    # torch.device("cpu")
    if isinstance(node, ast.Call):
        func_name = _dotted_name(node.func)
        if func_name in {"torch.device", "device"} and node.args:
            first = node.args[0]
            return (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value == "cpu"
            )
    return False


def _call_is_to_cpu(node: ast.Call) -> bool:
    """Return True if a .to(...) call targets CPU (simple literal cases)."""
    if node.args and _is_cpu_device_expr(node.args[0]):
        return True
    for kw in node.keywords:
        if kw.arg == "device" and kw.value is not None and _is_cpu_device_expr(kw.value):
            return True
    return False


class SyncVisitor(ast.NodeVisitor):
    """AST visitor that finds potential GPU synchronization points."""

    def __init__(self, path: str, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.hits: list[SyncHit] = []
        self._scope_stack: list[str] = []
        self._occurrences: dict[tuple[str, str], int] = {}

    def _current_scope(self) -> str:
        return ".".join(self._scope_stack) if self._scope_stack else "<module>"

    def _get_code_snippet(self, node: ast.AST) -> str:
        if hasattr(node, "lineno"):
            line_idx = node.lineno - 1
            if 0 <= line_idx < len(self.source_lines):
                return str(self.source_lines[line_idx]).strip()
        return "<unavailable>"

    def _record_hit(self, node: ast.AST, pattern: str) -> None:
        if pattern not in SYNC_METHOD_PATTERNS:
            raise ValueError(f"Unknown sync pattern: {pattern}")

        scope = self._current_scope()
        key = (scope, pattern)
        self._occurrences[key] = self._occurrences.get(key, 0) + 1
        occurrence = self._occurrences[key]

        self.hits.append(
            SyncHit(
                path=self.path,
                pattern=pattern,
                scope=scope,
                line=getattr(node, "lineno", 0),
                code_snippet=self._get_code_snippet(node),
                occurrence=occurrence,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr

            if attr in {"item", "tolist", "cpu", "numpy"}:
                self._record_hit(node, attr)
            elif attr == "to" and _call_is_to_cpu(node):
                self._record_hit(node, "to_cpu")
            elif attr == "synchronize":
                dotted = _dotted_name(node.func)
                # torch.cuda.synchronize(...) is explicit global sync; keep distinct.
                if dotted is not None and dotted.endswith("cuda.synchronize"):
                    self._record_hit(node, "cuda_synchronize")
                else:
                    self._record_hit(node, "synchronize")

        self.generic_visit(node)


class Whitelist:
    """Manages the GPU sync whitelist."""

    def __init__(self, config_path: Path) -> None:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        self.version = data.get("version", 1)
        self.allow_paths: list[str] = [
            p["path"] for p in data.get("allow_paths", [])
        ]
        raw_allow_hits = data.get("allow_hits", [])
        self.allow_hits: dict[str, dict[str, object]] = {}
        for entry in raw_allow_hits:
            key = entry["key"]
            parts = key.split(":")
            if len(parts) not in (3, 4):
                raise ValueError(
                    f"Invalid whitelist key format (expected path:scope:pattern[:occurrence]): {key}"
                )
            pattern = parts[2]
            if pattern not in SYNC_METHOD_PATTERNS:
                raise ValueError(f"Unknown pattern '{pattern}' in whitelist key: {key}")
            if len(parts) == 4:
                try:
                    int(parts[3])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid occurrence suffix in whitelist key: {key}"
                    ) from exc
            self.allow_hits[key] = entry
        self.today = date.today()

    def match(self, hit: SyncHit) -> tuple[bool, str | None, str | None]:
        # Broad path allowances (glob patterns)
        for pattern in self.allow_paths:
            if fnmatch(hit.path, pattern):
                return True, None, None

        # Exact key match (with optional occurrence fallback)
        keys_to_try = [hit.key]
        if hit.occurrence > 1:
            base_key = f"{hit.path}:{hit.scope}:{hit.pattern}"
            keys_to_try.append(base_key)

        for key in keys_to_try:
            entry = self.allow_hits.get(key)
            if entry is None:
                continue
            if "expires" in entry:
                expires = date.fromisoformat(str(entry["expires"]))
                if expires < self.today:
                    return False, f"EXPIRED: {key} (was {entry['expires']})", key
            return True, None, key

        return False, None, None


def find_sync_points(path: Path) -> list[SyncHit]:
    """Find all potential GPU sync points in a Python file."""
    try:
        source = path.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"WARNING: Could not parse {path}: {e}", file=sys.stderr)
        return []

    source_lines = source.splitlines()
    visitor = SyncVisitor(str(path), source_lines)
    visitor.visit(tree)
    return visitor.hits


def _generate_baseline_yaml(hits: list[SyncHit]) -> str:
    """Generate a baseline whitelist YAML for the current codebase."""
    allow_hits = []
    for hit in sorted(hits, key=lambda h: h.key):
        allow_hits.append(
            {
                "key": hit.key,
                "owner": "John",
                "reason": "grandfathered at initiation",
            }
        )

    header = (
        "# GPU Synchronization Whitelist\n"
        "# =============================\n"
        "#\n"
        "# This codebase treats CPU↔GPU synchronization points as performance hazards,\n"
        "# especially in Simic/Tolaria hot paths. All potential sync points must be\n"
        "# explicitly allowed here (or via allow_paths) so new syncs cannot slip into CI.\n"
        "#\n"
        "# Key format: \"path:scope:pattern[:occurrence]\"\n"
        "#   - path: repo-relative path\n"
        "#   - scope: qualname-ish scope (<module>, func, Class.method, etc)\n"
        "#   - pattern: item | tolist | cpu | numpy | synchronize | to_cpu | cuda_synchronize\n"
        "#   - occurrence: optional 1-based index if multiple in the same scope\n"
        "#\n"
        "# Generate stubs with:\n"
        "#   python scripts/lint_gpu_sync.py --generate-keys\n"
        "#\n"
    )

    doc = {
        "version": 1,
        "allow_paths": [
            {"path": "tests/**", "reason": "Tests may synchronize for assertions"},
            {"path": "scripts/**", "reason": "Utility/profiling scripts may synchronize"},
        ],
        "allow_hits": allow_hits,
    }
    # Keep stable ordering for diffs.
    return header + yaml.safe_dump(doc, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce GPU sync whitelist")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show allowed hits"
    )
    parser.add_argument(
        "--generate-keys",
        action="store_true",
        help="Print YAML allow_hits stubs for all detected sync points",
    )
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Print a complete baseline gpu_sync_whitelist.yaml for the current tree",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("gpu_sync_whitelist.yaml"),
        help="Path to whitelist config",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("src/esper"),
        help="Path to scan (default: src/esper)",
    )
    args = parser.parse_args()

    scan_path: Path = args.path
    hits: list[SyncHit] = []
    checked_files = 0

    for path in sorted(scan_path.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        checked_files += 1
        hits.extend(find_sync_points(path))

    if args.generate_keys:
        for hit in sorted(hits, key=lambda h: h.key):
            print(f"  - key: \"{hit.key}\"")
            print("    owner: \"\"")
            print(f"    reason: \"\"  # {hit.code_snippet[:80]}")
            print()
        print(f"\n# Found {len(hits)} sync point(s) in {checked_files} file(s)")
        return 0

    if args.generate_baseline:
        print(_generate_baseline_yaml(hits), end="")
        return 0

    config_path: Path = args.config
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return 1

    whitelist = Whitelist(config_path)
    violations: list[tuple[SyncHit, str | None]] = []
    warnings: list[str] = []
    allowed_hits = 0
    used_allow_hit_keys: set[str] = set()

    for hit in hits:
        allowed, message, matched_key = whitelist.match(hit)
        if message and "EXPIRED" in message:
            warnings.append(message)
        if not allowed:
            violations.append((hit, message))
        else:
            allowed_hits += 1
            if matched_key is not None:
                used_allow_hit_keys.add(matched_key)
            if args.verbose:
                print(f"OK: {hit.key} ({hit.code_snippet[:60]}...)")

    stale_allow_hits = sorted(set(whitelist.allow_hits.keys()) - used_allow_hit_keys)

    print(f"\nChecked {checked_files} file(s)")
    print(f"  Total sync points found: {len(hits)}")
    print(f"  Allowed: {allowed_hits}")
    print(f"  Violations: {len(violations)}")
    print(f"  Stale whitelist entries: {len(stale_allow_hits)}")

    for w in warnings:
        print(f"WARNING: {w}")

    if stale_allow_hits:
        print("\nSTALE WHITELIST ENTRIES FOUND:\n")
        for key in stale_allow_hits:
            print(f"  {key}")
        print("\nTo fix: remove or update these keys in gpu_sync_whitelist.yaml")
        return 1

    if violations:
        print(f"\n{len(violations)} VIOLATION(S) FOUND:\n")
        for hit, message in violations:
            print(f"  {hit.path}:{hit.line}: {hit.pattern} in {hit.scope}")
            print(f"    Code: {hit.code_snippet[:100]}")
            print(f"    Key:  {hit.key}")
            if message:
                print(f"    Note: {message}")
            print()
        print("To fix:")
        print("  1. PREFERRED: Remove the sync point (keep hot paths GPU-native)")
        print("  2. If legitimate: Add the key to gpu_sync_whitelist.yaml with justification")
        print("\nRun with --generate-keys to output whitelist entries for all hits")
        return 1

    print("All GPU sync points are whitelisted or absent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
