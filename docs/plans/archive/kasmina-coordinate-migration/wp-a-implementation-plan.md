# WP-A: Checkpoint Forensics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Audit checkpoint contents to understand exactly what types need to be converted for PyTorch 2.9 `weights_only=True` compatibility.

**Architecture:** Create a diagnostic script that recursively inspects checkpoint contents, identifies non-primitive types, and documents the gap between current serialization and PyTorch 2.9 requirements. Uses iterative audit to find ALL unsafe types (not just the first).

**Tech Stack:** Python 3.13, PyTorch 2.9, standard library (dataclasses, enum, datetime, collections)

---

## Task 1: Create scripts directory if needed

**Files:**
- Create: `scripts/` (directory)

**Step 1: Check if scripts directory exists**

Run: `ls -la scripts/ 2>/dev/null || echo "Directory does not exist"`

**Step 2: Create directory if needed**

Run: `mkdir -p scripts`

**Step 3: Verify**

Run: `ls -la scripts/`
Expected: Empty directory exists

---

## Task 2: Write the checkpoint audit tool

**Files:**
- Create: `scripts/checkpoint_audit.py`

**Step 1: Create the audit script**

```python
#!/usr/bin/env python3
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
        from esper.simic.ppo import PPOAgent
    except ImportError as e:
        print(f"Cannot import PPOAgent: {e}")
        print("Make sure PYTHONPATH includes src/")
        sys.exit(1)

    # Create a minimal agent
    agent = PPOAgent(
        state_dim=50,  # Approximate real state dim
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
        from esper.simic.features import TaskConfig
    except ImportError as e:
        print(f"Cannot import Kasmina modules: {e}")
        print("Make sure PYTHONPATH includes src/")
        sys.exit(1)

    # Create model with seed slots
    host = CNNHost(num_classes=10, n_blocks=3)
    task_config = TaskConfig(topology="cnn", blending_steps=10)

    model = MorphogeneticModel(
        host=host,
        device="cpu",
        slots=["early", "mid", "late"],
        task_config=task_config,
    )

    # Germinate a seed to populate SeedState
    model.germinate_seed(
        blueprint_id="norm",
        seed_id="test-seed-001",
        slot="mid",
        blend_algorithm_id="sigmoid",
    )

    # Get state dict (includes extra_state from SeedSlot)
    state_dict = model.state_dict()

    # Save
    checkpoint_path = Path("/tmp/esper_morphogenetic_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": {"slots": ["early", "mid", "late"]},
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
        from esper.simic.features import TaskConfig
    except ImportError as e:
        print(f"Cannot import Kasmina modules: {e}")
        print("Make sure PYTHONPATH includes src/")
        sys.exit(1)

    # Create model with seed slots
    host = CNNHost(num_classes=10, n_blocks=3)
    task_config = TaskConfig(topology="cnn", blending_steps=10)

    model = MorphogeneticModel(
        host=host,
        device="cpu",
        slots=["early", "mid", "late"],
        task_config=task_config,
    )

    # Germinate a seed
    model.germinate_seed(
        blueprint_id="norm",
        seed_id="test-seed-blending",
        slot="mid",
        blend_algorithm_id="gated",  # Use gated to get nn.Module alpha_schedule
    )

    # Transition to TRAINING then BLENDING
    slot = model.slots["mid"]
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
            "config": {"slots": ["early", "mid", "late"]},
        },
        checkpoint_path,
    )

    print(f"Saved BLENDING checkpoint to: {checkpoint_path}")
    return checkpoint_path


def generate_probationary_checkpoint() -> Path:
    """Generate a checkpoint with PROBATIONARY state (after blending).

    This tests what state persists after blending completes.
    """
    print("Generating MorphogeneticModel checkpoint (PROBATIONARY state)...")
    print("=" * 60)

    try:
        from esper.kasmina.host import CNNHost, MorphogeneticModel
        from esper.leyline.stages import SeedStage
        from esper.simic.features import TaskConfig
    except ImportError as e:
        print(f"Cannot import Kasmina modules: {e}")
        print("Make sure PYTHONPATH includes src/")
        sys.exit(1)

    # Create model with seed slots
    host = CNNHost(num_classes=10, n_blocks=3)
    task_config = TaskConfig(topology="cnn", blending_steps=3)

    model = MorphogeneticModel(
        host=host,
        device="cpu",
        slots=["early", "mid", "late"],
        task_config=task_config,
    )

    # Germinate a seed
    model.germinate_seed(
        blueprint_id="norm",
        seed_id="test-seed-probationary",
        slot="mid",
        blend_algorithm_id="linear",
    )

    # Full lifecycle to PROBATIONARY
    slot = model.slots["mid"]
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.start_blending(total_steps=3)

    # Complete blending
    slot.state.alpha = 1.0  # Force alpha to completion threshold
    slot.state.metrics.epochs_in_current_stage = 5  # Meet minimum epochs
    slot.state.transition(SeedStage.PROBATIONARY)

    print(f"Stage: {slot.state.stage.name}")
    print(f"alpha_schedule: {slot.alpha_schedule}")

    # Get state dict
    state_dict = model.state_dict()

    # Save
    checkpoint_path = Path("/tmp/esper_probationary_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": {"slots": ["early", "mid", "late"]},
        },
        checkpoint_path,
    )

    print(f"Saved PROBATIONARY checkpoint to: {checkpoint_path}")
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
    elif arg == "--generate-probationary":
        # Generate and audit checkpoint with PROBATIONARY state
        path = generate_probationary_checkpoint()
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
```

**Step 2: Make the script executable**

Run: `chmod +x scripts/checkpoint_audit.py`

**Step 3: Verify syntax**

Run: `python -m py_compile scripts/checkpoint_audit.py && echo "Syntax OK"`
Expected: `Syntax OK`

**Step 4: Commit**

```bash
git add scripts/checkpoint_audit.py
git commit -m "feat(scripts): add checkpoint audit tool for PyTorch 2.9 compatibility

Recursively inspects checkpoint contents to identify types that are
incompatible with weights_only=True loading. Supports:
- Auditing existing checkpoints
- Generating test checkpoints for multiple lifecycle states
- Iterative audit to find ALL unsafe types (not just first)
- Optimizer state auditing
- Environment info capture (PyTorch, CUDA, Python versions)"
```

---

## Task 3: Generate and audit PPO checkpoint

**Files:**
- None (diagnostic task)

**Step 1: Generate PPO test checkpoint**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-test
```

Expected output pattern:
```
Generating test checkpoint...
==================================================
Saved test checkpoint to: /tmp/esper_test_checkpoint.pt

Auditing: /tmp/esper_test_checkpoint.pt
==================================================

0. Environment
==============
  PyTorch: 2.x.x
  ...

1. Testing weights_only=True
=============================
FAILED: ...
```

**Step 2: Capture output to file**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-test 2>&1 | tee /tmp/ppo_audit_output.txt
```

**Step 3: Review findings**

Run: `cat /tmp/ppo_audit_output.txt`

Document any issues found in the output.

---

## Task 4: Generate and audit MorphogeneticModel checkpoint (GERMINATED)

**Files:**
- None (diagnostic task)

**Step 1: Generate MorphogeneticModel checkpoint**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-morphogenetic
```

Expected: This exercises `SeedSlot.get_extra_state()` which stores:
- `SeedState` dataclass
- `alpha_schedule` (None at GERMINATED)
- `isolate_gradients` (bool)

**Step 2: Capture output to file**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-morphogenetic 2>&1 | tee /tmp/morphogenetic_audit_output.txt
```

---

## Task 5: Generate and audit BLENDING checkpoint (critical for nn.Module issues)

**Files:**
- None (diagnostic task)

**Step 1: Generate BLENDING checkpoint**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-blending
```

Expected: This exercises `SeedSlot.get_extra_state()` with:
- `SeedState` dataclass (with populated stage_history)
- `alpha_schedule` (GatedBlend nn.Module - **THIS IS THE CRITICAL ISSUE**)
- `isolate_gradients` (bool)

**Step 2: Capture output to file**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-blending 2>&1 | tee /tmp/blending_audit_output.txt
```

**Step 3: Review nn.Module issues**

Run: `grep -i "nn.Module" /tmp/blending_audit_output.txt`

This should surface the `alpha_schedule` serialization problem.

---

## Task 6: Generate and audit PROBATIONARY checkpoint

**Files:**
- None (diagnostic task)

**Step 1: Generate PROBATIONARY checkpoint**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-probationary
```

Expected: This shows what persists after blending completes.

**Step 2: Capture output to file**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-probationary 2>&1 | tee /tmp/probationary_audit_output.txt
```

---

## Task 7: Run iterative audit on BLENDING checkpoint

**Files:**
- None (diagnostic task)

**Step 1: Run iterative audit**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --iterative /tmp/esper_blending_checkpoint.pt 2>&1 | tee /tmp/iterative_audit_output.txt
```

Expected: Lists ALL unsafe types, not just the first one encountered.

**Step 2: Review all unsafe types**

Run: `cat /tmp/iterative_audit_output.txt`

Document the complete list of types that need conversion.

---

## Task 8: Document audit findings

**Files:**
- Create: `docs/plans/kasmina-coordinate-migration/checkpoint-audit-results.md`

**Step 1: Create findings document**

Based on the audit outputs from Tasks 3-7, create the results document:

```markdown
# Checkpoint Audit Results

**Date:** 2025-12-16
**Auditor:** [Your name]

---

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | [from audit output] |
| CUDA | [from audit output] |
| Python | [from audit output] |
| Platform | [from audit output] |

---

## Summary

| Checkpoint Type | Stage | weights_only=True | Issues Found |
|-----------------|-------|-------------------|--------------|
| PPO Agent | N/A | PASS/FAIL | N issues |
| MorphogeneticModel | GERMINATED | PASS/FAIL | N issues |
| MorphogeneticModel | BLENDING | PASS/FAIL | N issues |
| MorphogeneticModel | PROBATIONARY | PASS/FAIL | N issues |

---

## PPO Agent Checkpoint

**Generated:** `/tmp/esper_test_checkpoint.pt`

### weights_only=True Result

```
[Paste exact error message]
```

### Issues Found

[List each issue from audit output]

### Top-Level Structure

[Paste top-level keys section]

---

## MorphogeneticModel Checkpoint (GERMINATED)

**Generated:** `/tmp/esper_morphogenetic_checkpoint.pt`

### Issues Found

[List issues - should be fewer than BLENDING since no alpha_schedule]

---

## MorphogeneticModel Checkpoint (BLENDING) - CRITICAL

**Generated:** `/tmp/esper_blending_checkpoint.pt`

### weights_only=True Result

```
[Paste exact error message - likely mentions GatedBlend]
```

### Issues Found

[List each issue - should include nn.Module for alpha_schedule]

### Critical Finding: alpha_schedule

The `alpha_schedule` field stores an `nn.Module` (e.g., `GatedBlend`), which cannot be serialized with `weights_only=True`.

**Current behavior:**
```python
# SeedSlot.get_extra_state()
return {
    "seed_state": self.state,           # SeedState dataclass
    "alpha_schedule": self.alpha_schedule,  # nn.Module - FAILS
    "isolate_gradients": self.isolate_gradients,
}
```

**Required fix for M4:**
- Discard `alpha_schedule` after BLENDING completes
- Or serialize only config, not the nn.Module itself

---

## MorphogeneticModel Checkpoint (PROBATIONARY)

**Generated:** `/tmp/esper_probationary_checkpoint.pt`

### Issues Found

[Document whether alpha_schedule persists after BLENDING]

---

## Iterative Audit Results

**All unsafe types found:**

```
[Paste from iterative audit output]
```

---

## Type Conversion Requirements for M4

| Current Type | Location | Convert To | Priority |
|--------------|----------|------------|----------|
| `SeedState` dataclass | `extra_state['seed_state']` | `dict` via `to_dict()` | High |
| `SeedStage` Enum | `SeedState.stage` | `int` (enum.value) | High |
| `datetime` | `SeedState.stage_entered_at` | ISO 8601 string | High |
| `deque` | `SeedState.stage_history` | `list` | High |
| `GatedBlend` nn.Module | `extra_state['alpha_schedule']` | Config dict OR discard | Critical |
| `SeedTelemetry` dataclass | `SeedState.telemetry` | `dict` via `to_dict()` | Medium |

---

## Recommended Changes for M4

### 1. SeedSlot.get_extra_state() - CRITICAL

```python
def get_extra_state(self) -> dict:
    """Persist SeedState for PyTorch 2.9+ weights_only=True compatibility."""
    state_dict = {
        "isolate_gradients": self.isolate_gradients,
    }

    if self.state is not None:
        state_dict["seed_state"] = self.state.to_dict()

    # Alpha schedule: serialize config only, NOT the nn.Module
    if self.alpha_schedule is not None:
        state_dict["alpha_schedule_config"] = {
            "algorithm_id": getattr(self.alpha_schedule, "algorithm_id", None),
            "total_steps": getattr(self.alpha_schedule, "total_steps", None),
            # GatedBlend weights saved in state_dict(), not here
        }

    return state_dict
```

### 2. SeedState.to_dict() / from_dict()

```python
def to_dict(self) -> dict:
    """Convert to primitive dict for serialization."""
    return {
        "seed_id": self.seed_id,
        "blueprint_id": self.blueprint_id,
        "slot_id": self.slot_id,
        "stage": self.stage.value,  # Enum -> int
        "stage_entered_at": self.stage_entered_at.isoformat(),  # datetime -> str
        "alpha": self.alpha,
        "stage_history": list(self.stage_history),  # deque -> list
        # ... other fields
    }

@classmethod
def from_dict(cls, data: dict) -> "SeedState":
    """Reconstruct from primitive dict."""
    return cls(
        seed_id=data["seed_id"],
        stage=SeedStage(data["stage"]),  # int -> Enum
        stage_entered_at=datetime.fromisoformat(data["stage_entered_at"]),
        stage_history=deque(data["stage_history"]),
        # ...
    )
```

### 3. Discard alpha_schedule after BLENDING

```python
# In SeedSlot, after BLENDING -> PROBATIONARY transition:
if target_stage == SeedStage.PROBATIONARY:
    self.alpha_schedule = None  # No longer needed
    self.state.alpha = 1.0  # Permanent full blend
```

---

## Verification

After M4 implementation, re-run audits to confirm:
```bash
PYTHONPATH=src python scripts/checkpoint_audit.py --generate-blending
```

Expected: `SUCCESS - checkpoint is already compatible!`
```

**Step 2: Commit findings document**

```bash
git add docs/plans/kasmina-coordinate-migration/checkpoint-audit-results.md
git commit -m "docs: add checkpoint audit results for M4 planning

Documents PyTorch 2.9 weights_only=True compatibility issues found in:
- PPO Agent checkpoints
- MorphogeneticModel at GERMINATED, BLENDING, PROBATIONARY stages

Critical finding: alpha_schedule (nn.Module) must be discarded or
serialized as config only. Includes type conversion requirements."
```

---

## Task 9: Final verification

**Step 1: Verify all outputs exist**

Run:
```bash
ls -la scripts/checkpoint_audit.py
ls -la docs/plans/kasmina-coordinate-migration/checkpoint-audit-results.md
```

Expected: Both files exist

**Step 2: Run audit tool help**

Run: `PYTHONPATH=src uv run python scripts/checkpoint_audit.py --help`

Expected: Help text displays all options including --generate-blending and --iterative

**Step 3: Final commit (if any uncommitted changes)**

Run: `git status`

If clean, WP-A is complete.

---

## Acceptance Checklist

- [ ] `scripts/checkpoint_audit.py` exists and runs
- [ ] PPO checkpoint audited, output captured
- [ ] MorphogeneticModel GERMINATED checkpoint audited
- [ ] MorphogeneticModel BLENDING checkpoint audited (critical for nn.Module)
- [ ] MorphogeneticModel PROBATIONARY checkpoint audited
- [ ] Iterative audit run to find ALL unsafe types
- [ ] `checkpoint-audit-results.md` documents all findings
- [ ] Type conversion requirements clearly specified
- [ ] All changes committed

---

## Outputs

1. **`scripts/checkpoint_audit.py`** — Reusable diagnostic tool with:
   - Environment info capture
   - Optimizer state auditing
   - Multiple lifecycle state generators
   - Iterative audit mode
2. **`docs/plans/kasmina-coordinate-migration/checkpoint-audit-results.md`** — Findings document

These outputs directly inform M4 implementation scope.
