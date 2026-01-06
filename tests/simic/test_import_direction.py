"""Import direction guardrails for Phase 3 module splits."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _module_path_for(file_path: Path, src_root: Path) -> str:
    rel = file_path.relative_to(src_root).with_suffix("")
    return ".".join(rel.parts)


def _resolve_import(module_path: str, node: ast.ImportFrom) -> str | None:
    if node.module is None:
        return None
    if node.level == 0:
        return node.module
    base_parts = module_path.split(".")
    if node.level > len(base_parts):
        return None
    base_parts = base_parts[: -node.level]
    if node.module:
        base_parts += node.module.split(".")
    return ".".join(base_parts)


def _collect_imports(file_path: Path, src_root: Path) -> set[str]:
    module_path = _module_path_for(file_path, src_root)
    tree = ast.parse(file_path.read_text())
    imports: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_import(module_path, node)
            if resolved is not None:
                imports.add(resolved)

    return imports


def _assert_no_prefix_imports(file_path: Path, src_root: Path, banned_prefixes: tuple[str, ...]) -> None:
    imports = _collect_imports(file_path, src_root)
    for imported in sorted(imports):
        for banned in banned_prefixes:
            if imported == banned or imported.startswith(f"{banned}."):
                raise AssertionError(f"{file_path} imports {imported}, blocked by {banned}")


def test_rewards_do_not_import_training() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    rewards_root = src_root / "esper" / "simic" / "rewards"

    for path in rewards_root.rglob("*.py"):
        _assert_no_prefix_imports(path, src_root, ("esper.simic.training",))


def test_telemetry_does_not_import_training_or_ppo_update() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    telemetry_root = src_root / "esper" / "simic" / "telemetry"

    for path in telemetry_root.rglob("*.py"):
        _assert_no_prefix_imports(
            path,
            src_root,
            ("esper.simic.training", "esper.simic.agent.ppo_update"),
        )


def test_ppo_update_does_not_import_telemetry() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    ppo_update_path = src_root / "esper" / "simic" / "agent" / "ppo_update.py"

    if not ppo_update_path.exists():
        pytest.skip("ppo_update.py not created yet")

    _assert_no_prefix_imports(
        ppo_update_path,
        src_root,
        ("esper.simic.telemetry",),
    )


def test_reward_internals_not_imported_outside_rewards() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    rewards_root = src_root / "esper" / "simic" / "rewards"

    internal_modules = (
        "esper.simic.rewards.contribution",
        "esper.simic.rewards.loss_primary",
        "esper.simic.rewards.shaping",
    )
    existing_internals = [
        mod for mod in internal_modules if (rewards_root / f"{mod.split('.')[-1]}.py").exists()
    ]
    if not existing_internals:
        pytest.skip("reward internals not split yet")

    for path in src_root.rglob("*.py"):
        if rewards_root in path.parents:
            continue
        imports = _collect_imports(path, src_root)
        for imported in sorted(imports):
            for banned in existing_internals:
                if imported == banned or imported.startswith(f"{banned}."):
                    raise AssertionError(f"{path} imports internal reward module {imported}")
