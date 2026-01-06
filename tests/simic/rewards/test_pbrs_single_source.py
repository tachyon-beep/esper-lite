"""Guardrail: PBRS potentials must have a single definition site."""

from __future__ import annotations

import ast
from pathlib import Path


def _count_stage_potentials_assignments(path: Path) -> int:
    tree = ast.parse(path.read_text())
    count = 0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STAGE_POTENTIALS":
                    count += 1
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "STAGE_POTENTIALS":
                count += 1
    return count


def test_stage_potentials_single_definition() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    rewards_root = repo_root / "src" / "esper" / "simic" / "rewards"

    assignments = {}
    for path in rewards_root.rglob("*.py"):
        count = _count_stage_potentials_assignments(path)
        if count:
            assignments[path] = count

    total = sum(assignments.values())
    assert total == 1, f"Expected 1 STAGE_POTENTIALS definition, found {total}: {assignments}"
