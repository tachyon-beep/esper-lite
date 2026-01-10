#!/usr/bin/env python3
# mypy: ignore-errors
"""Audit checkpoint contents for PyTorch 2.9 compatibility.

PyTorch 2.9 defaults to weights_only=True in torch.load(), which rejects
arbitrary Python objects. This script identifies non-primitive types in
checkpoints that need conversion.

Features:
- Iterative audit to find ALL unsafe types (not just the first)
- Environment info capture (PyTorch, CUDA, Python versions)
- Optimizer state auditing
- Multiple checkpoint generators for different lifecycle states

Usage:
    python scripts/checkpoint_audit.py <checkpoint.pt>
    python scripts/checkpoint_audit.py --generate-test
    python scripts/checkpoint_audit.py --generate-morphogenetic
    python scripts/checkpoint_audit.py --generate-blending
    python scripts/checkpoint_audit.py --generate-holding
    python scripts/checkpoint_audit.py --iterative <checkpoint.pt>

Examples:
    # Audit an existing checkpoint
    python scripts/checkpoint_audit.py checkpoints/agent.pt

    # Generate a test checkpoint and audit it
    python scripts/checkpoint_audit.py --generate-test

    # Find ALL unsafe types iteratively
    python scripts/checkpoint_audit.py --iterative checkpoints/agent.pt
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict, deque
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def get_environment_info() -> dict[str, Any]:
    """Capture comprehensive environment information for audit report."""
    return {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "python_version": sys.version,
        "platform": sys.platform,
    }


# Types that are safe for weights_only=True loading
# Based on PyTorch 2.9's actual allowlist
SAFE_TYPES = (
    # Python primitives
    int,
    float,
    str,
    bool,
    type(None),
    bytes,
    # Collections
    list,
    tuple,
    dict,
    set,
    frozenset,
    OrderedDict,
    # PyTorch types
    torch.Tensor,
    torch.dtype,
    torch.device,
    torch.Size,
    torch.storage.TypedStorage,
    # Note: torch.nn.parameter.Parameter is handled within state_dict context
)


def audit_value(
    value: Any,
    path: str = "root",
    issues: list[str] | None = None,
    depth: int = 0,
    max_depth: int = 20,
) -> list[str]:
    """Recursively audit a value for non-primitive types.

    Args:
        value: The value to audit
        path: Dot-notation path for reporting (e.g., "root['key'][0]")
        issues: Accumulated list of issues (created if None)
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        List of issue strings describing non-compatible types
    """
    if issues is None:
        issues = []

    if depth > max_depth:
        issues.append(f"{path}: MAX DEPTH EXCEEDED (possible circular reference)")
        return issues

    value_type = type(value)

    # Check for known problematic types (most specific first)
    if isinstance(value, datetime):
        issues.append(f"{path}: datetime.datetime -> convert to ISO 8601 string")
    elif isinstance(value, deque):
        issues.append(f"{path}: collections.deque -> convert to list")
        # Still recurse into deque contents
        for i, v in enumerate(value):
            audit_value(v, f"{path}[{i}]", issues, depth + 1, max_depth)
    elif isinstance(value, Enum):
        issues.append(
            f"{path}: Enum {value_type.__module__}.{value_type.__name__}.{value.name} "
            f"-> convert to int (enum.value) for forward compatibility"
        )
    elif is_dataclass(value) and not isinstance(value, type):
        issues.append(
            f"{path}: dataclass {value_type.__module__}.{value_type.__name__} "
            f"-> convert to dict via to_dict() method"
        )
        # Recurse into dataclass fields
        for field_name in value.__dataclass_fields__:
            field_value = getattr(value, field_name)
            audit_value(field_value, f"{path}.{field_name}", issues, depth + 1, max_depth)
    elif isinstance(value, nn.Module):
        issues.append(
            f"{path}: nn.Module {value_type.__module__}.{value_type.__name__} "
            f"-> CRITICAL: store state_dict separately, serialize config only"
        )
    elif isinstance(value, (dict, OrderedDict)):
        for k, v in value.items():
            key_repr = repr(k) if not isinstance(k, str) else k
            audit_value(v, f"{path}['{key_repr}']", issues, depth + 1, max_depth)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            audit_value(v, f"{path}[{i}]", issues, depth + 1, max_depth)
    elif not isinstance(value, SAFE_TYPES):
        issues.append(
            f"{path}: UNKNOWN TYPE {value_type.__module__}.{value_type.__name__} "
            f"-> needs manual inspection"
        )

    return issues


def print_section(title: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{title}")
    print(char * len(title))


def iterative_audit(checkpoint_path: Path) -> list[str]:
    """Find ALL unsafe types by iteratively adding them to allowlist.

    PyTorch's weights_only=True fails on first unsafe type encountered.
    This function iteratively discovers all unsafe types by:
    1. Attempting load with weights_only=True
    2. Parsing error for offending type
    3. Adding to temporary allowlist
    4. Retrying until success or max iterations

    Returns:
        List of all type names that needed allowlisting
    """
    print_section("Iterative Audit Mode")
    print("Finding ALL unsafe types (not just the first)...")

    unsafe_types: list[str] = []
    max_iterations = 50  # Safety limit

    # Pattern to extract type from PyTorch error message
    # Example: "Unsupported class esper.leyline.stages.SeedStage"
    type_pattern = re.compile(r"Unsupported class (\S+)")

    for iteration in range(max_iterations):
        try:
            # Try loading with current allowlist
            torch.load(checkpoint_path, weights_only=True)
            print(f"\nSuccess after {iteration} type additions!")
            break
        except Exception as e:
            error_msg = str(e)
            match = type_pattern.search(error_msg)

            if match:
                type_name = match.group(1)
                if type_name not in unsafe_types:
                    unsafe_types.append(type_name)
                    print(f"  [{iteration + 1}] Found: {type_name}")

                    # Try to add to safe globals (PyTorch 2.9+)
                    try:
                        # Import the type dynamically
                        module_path, class_name = type_name.rsplit(".", 1)
                        module = __import__(module_path, fromlist=[class_name])
                        cls = getattr(module, class_name)
                        torch.serialization.add_safe_globals([cls])
                    except Exception:
                        # Can't add - just document it
                        pass
                else:
                    # Same type again - likely a deeper issue
                    print(f"  [{iteration + 1}] Stuck on: {type_name}")
                    break
            else:
                # Different error format - stop
                print(f"  [{iteration + 1}] Unexpected error: {error_msg[:100]}...")
                break
    else:
        print(f"\nReached iteration limit ({max_iterations})")

    return unsafe_types


def audit_checkpoint(checkpoint_path: Path, include_optimizer: bool = True) -> dict[str, Any]:
    """Audit a checkpoint file and return results.

    Args:
        checkpoint_path: Path to the checkpoint file
        include_optimizer: Whether to audit optimizer state (default True)

    Returns:
        Dict with audit results including issues, top_level_keys, and error info
    """
    results: dict[str, Any] = {
        "path": str(checkpoint_path),
        "environment": get_environment_info(),
        "weights_only_compatible": False,
        "weights_only_error": None,
        "issues": [],
        "top_level_keys": {},
    }

    print(f"Auditing: {checkpoint_path}")
    print("=" * 60)

    # Print environment info
    print_section("0. Environment")
    env = results["environment"]
    print(f"  PyTorch: {env['pytorch_version']}")
    print(f"  CUDA: {env['cuda_version'] or 'Not available'}")
    print(f"  Python: {env['python_version'].split()[0]}")
    print(f"  Platform: {env['platform']}")

    # Step 1: Test with weights_only=True
    print_section("1. Testing weights_only=True")
    try:
        torch.load(checkpoint_path, weights_only=True)
        print("SUCCESS - checkpoint is already compatible!")
        results["weights_only_compatible"] = True
        return results
    except Exception as e:
        error_msg = str(e)
        print(f"FAILED: {error_msg[:200]}...")
        results["weights_only_error"] = error_msg

    # Step 2: Load with weights_only=False and analyze
    print_section("2. Loading with weights_only=False")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as e:
        print(f"FAILED to load checkpoint: {e}")
        results["load_error"] = str(e)
        return results

    print(f"Loaded successfully. Type: {type(checkpoint).__name__}")

    # Step 3: Audit contents
    print_section("3. Auditing contents for non-primitive types")
    issues = audit_value(checkpoint)
    results["issues"] = issues

    if issues:
        print(f"Found {len(issues)} compatibility issues:\n")
        for issue in issues:
            print(f"  * {issue}")
    else:
        print("No issues found (but weights_only=True still failed?)")
        print("This may indicate a PyTorch version mismatch or custom unpickler issue.")

    # Step 4: Audit optimizer state if present and requested
    if include_optimizer and isinstance(checkpoint, dict):
        if "optimizer_state_dict" in checkpoint:
            print_section("3b. Auditing optimizer state")
            opt_issues = audit_value(
                checkpoint["optimizer_state_dict"],
                path="optimizer_state_dict"
            )
            if opt_issues:
                print(f"Found {len(opt_issues)} optimizer state issues:\n")
                for issue in opt_issues:
                    print(f"  * {issue}")
                results["issues"].extend(opt_issues)
            else:
                print("Optimizer state is clean.")
        else:
            print_section("3b. Optimizer State")
            print("No optimizer_state_dict found in checkpoint.")

    # Step 5: Show top-level structure
    print_section("4. Top-level checkpoint structure")
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            type_name = type(value).__name__
            if isinstance(value, dict):
                type_name = f"dict[{len(value)} keys]"
            elif isinstance(value, (list, tuple)):
                type_name = f"{type(value).__name__}[{len(value)} items]"
            elif isinstance(value, torch.Tensor):
                type_name = f"Tensor{list(value.shape)}"
            print(f"  * {key}: {type_name}")
            results["top_level_keys"][key] = type_name
    else:
        print(f"  (not a dict, is {type(checkpoint).__name__})")

    return results


def generate_test_checkpoint() -> Path:
    """Generate a test checkpoint for auditing.

    Creates a minimal PPO training run to produce a checkpoint with
    realistic SeedSlot extra_state contents.
    """
    print("Generating test checkpoint...")
    print("=" * 60)

    # Import here to avoid dependency if just auditing
    try:
        from esper.leyline.slot_config import SlotConfig
        from esper.simic.agent import PPOAgent
        from esper.tamiyo.policy.factory import create_policy
        from esper.tamiyo.policy.features import get_feature_size
    except ImportError as e:
        print(f"Cannot import required modules: {e}")
        print("Make sure PYTHONPATH includes src/")
        sys.exit(1)

    # Create a minimal agent with PolicyBundle
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        state_dim=get_feature_size(slot_config),
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        device="cpu",
    )

    # Save to temp location
    checkpoint_path = Path("/tmp/esper_test_checkpoint.pt")
    agent.save(checkpoint_path, metadata={"test": True, "timestamp": "2025-12-16"})

    print(f"Saved test checkpoint to: {checkpoint_path}")
    return checkpoint_path


def generate_morphogenetic_checkpoint() -> Path:
    """Generate a checkpoint with MorphogeneticModel and SeedSlot state.

    This exercises the full extra_state serialization path with GERMINATED state.
    """
    print("Generating MorphogeneticModel checkpoint (GERMINATED state)...")
    print("=" * 60)

    try:
        from esper.kasmina.host import CNNHost, MorphogeneticModel
        from esper.tamiyo.policy.features import TaskConfig
    except ImportError as e:
        print(f"Cannot import Kasmina modules: {e}")
        print("Make sure PYTHONPATH includes src/")
        sys.exit(1)

    # Create model with seed slots
    host = CNNHost(num_classes=10, n_blocks=3)
    task_config = TaskConfig(
        task_type="classification",
        topology="cnn",
        baseline_loss=2.3,
        target_loss=0.3,
        typical_loss_delta_std=0.05,
        max_epochs=25,
        blending_steps=10,
    )

    model = MorphogeneticModel(
        host=host,
        device="cpu",
        slots=["r0c0", "r0c1", "r0c2"],
        task_config=task_config,
    )

    # Germinate a seed to populate SeedState
    model.germinate_seed(
        blueprint_id="norm",
        seed_id="test-seed-001",
        slot="r0c1",
        blend_algorithm_id="sigmoid",
    )

    # Get state dict (includes extra_state from SeedSlot)
    state_dict = model.state_dict()

    # Save
    checkpoint_path = Path("/tmp/esper_morphogenetic_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": {"slots": ["r0c0", "r0c1", "r0c2"]},
        },
        checkpoint_path,
    )

    print(f"Saved MorphogeneticModel checkpoint to: {checkpoint_path}")
    return checkpoint_path


def generate_blending_checkpoint() -> Path:
    """Generate a checkpoint with active BLENDING state.

    This exercises SeedSlot with alpha_schedule (nn.Module) populated.
    Critical for finding nn.Module serialization issues.
    """
    print("Generating MorphogeneticModel checkpoint (BLENDING state)...")
    print("=" * 60)

    try:
        from esper.kasmina.host import CNNHost, MorphogeneticModel
        from esper.leyline.stages import SeedStage
        from esper.tamiyo.policy.features import TaskConfig
    except ImportError as e:
        print(f"Cannot import Kasmina modules: {e}")
        print("Make sure PYTHONPATH includes src/")
        sys.exit(1)

    # Create model with seed slots
    host = CNNHost(num_classes=10, n_blocks=3)
    task_config = TaskConfig(
        task_type="classification",
        topology="cnn",
        baseline_loss=2.3,
        target_loss=0.3,
        typical_loss_delta_std=0.05,
        max_epochs=25,
        blending_steps=10,
    )

    model = MorphogeneticModel(
        host=host,
        device="cpu",
        slots=["r0c0", "r0c1", "r0c2"],
        task_config=task_config,
    )

    # Germinate a seed
    model.germinate_seed(
        blueprint_id="norm",
        seed_id="test-seed-blending",
        slot="r0c1",
        blend_algorithm_id="gated",  # Use gated to get nn.Module alpha_schedule
    )

    # Transition to TRAINING then BLENDING
    slot = model.seed_slots["r0c1"]
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.start_blending(total_steps=10)

    # Verify alpha_schedule is set (this is what causes serialization issues)
    assert slot.alpha_schedule is not None, "alpha_schedule should be set"
    print(f"alpha_schedule type: {type(slot.alpha_schedule).__name__}")

    # Get state dict
    state_dict = model.state_dict()

    # Save
    checkpoint_path = Path("/tmp/esper_blending_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": {"slots": ["r0c0", "r0c1", "r0c2"]},
        },
        checkpoint_path,
    )

    print(f"Saved BLENDING checkpoint to: {checkpoint_path}")
    return checkpoint_path


def generate_holding_checkpoint() -> Path:
    """Generate a checkpoint with HOLDING state (after blending).

    This tests what state persists after blending completes.
    """
    print("Generating MorphogeneticModel checkpoint (HOLDING state)...")
    print("=" * 60)

    try:
        from esper.kasmina.host import CNNHost, MorphogeneticModel
        from esper.leyline.stages import SeedStage
        from esper.tamiyo.policy.features import TaskConfig
    except ImportError as e:
        print(f"Cannot import Kasmina modules: {e}")
        print("Make sure PYTHONPATH includes src/")
        sys.exit(1)

    # Create model with seed slots
    host = CNNHost(num_classes=10, n_blocks=3)
    task_config = TaskConfig(
        task_type="classification",
        topology="cnn",
        baseline_loss=2.3,
        target_loss=0.3,
        typical_loss_delta_std=0.05,
        max_epochs=25,
        blending_steps=3,
    )

    model = MorphogeneticModel(
        host=host,
        device="cpu",
        slots=["r0c0", "r0c1", "r0c2"],
        task_config=task_config,
    )

    # Germinate a seed
    model.germinate_seed(
        blueprint_id="norm",
        seed_id="test-seed-holding",
        slot="r0c1",
        blend_algorithm_id="linear",
    )

    # Full lifecycle to HOLDING
    slot = model.seed_slots["r0c1"]
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.start_blending(total_steps=3)

    # Complete blending
    slot.state.alpha = 1.0  # Force alpha to completion threshold
    slot.state.metrics.epochs_in_current_stage = 5  # Meet minimum epochs
    slot.state.transition(SeedStage.HOLDING)

    print(f"Stage: {slot.state.stage.name}")
    print(f"alpha_schedule: {slot.alpha_schedule}")

    # Get state dict
    state_dict = model.state_dict()

    # Save
    checkpoint_path = Path("/tmp/esper_holding_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": {"slots": ["r0c0", "r0c1", "r0c2"]},
        },
        checkpoint_path,
    )

    print(f"Saved HOLDING checkpoint to: {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--generate-test":
        # Generate and audit PPO checkpoint
        path = generate_test_checkpoint()
        print("\n")
        audit_checkpoint(path)
    elif arg == "--generate-morphogenetic":
        # Generate and audit MorphogeneticModel checkpoint (GERMINATED)
        path = generate_morphogenetic_checkpoint()
        print("\n")
        audit_checkpoint(path)
    elif arg == "--generate-blending":
        # Generate and audit checkpoint with BLENDING state
        path = generate_blending_checkpoint()
        print("\n")
        audit_checkpoint(path)
    elif arg == "--generate-holding":
        # Generate and audit checkpoint with HOLDING state
        path = generate_holding_checkpoint()
        print("\n")
        audit_checkpoint(path)
    elif arg == "--iterative":
        # Iterative audit to find ALL unsafe types
        if len(sys.argv) < 3:
            print("Usage: checkpoint_audit.py --iterative <checkpoint.pt>")
            sys.exit(1)
        path = Path(sys.argv[2])
        if not path.exists():
            print(f"Error: Checkpoint file not found: {path}")
            sys.exit(1)
        unsafe_types = iterative_audit(path)
        print("\n" + "=" * 60)
        print(f"TOTAL UNSAFE TYPES FOUND: {len(unsafe_types)}")
        for t in unsafe_types:
            print(f"  - {t}")
    elif arg == "--help" or arg == "-h":
        print(__doc__)
    else:
        # Audit provided checkpoint
        path = Path(arg)
        if not path.exists():
            print(f"Error: Checkpoint file not found: {path}")
            sys.exit(1)
        audit_checkpoint(path)


if __name__ == "__main__":
    main()
