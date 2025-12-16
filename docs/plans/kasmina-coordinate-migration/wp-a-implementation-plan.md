# WP-A: Checkpoint Forensics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Audit checkpoint contents to understand exactly what types need to be converted for PyTorch 2.9 `weights_only=True` compatibility.

**Architecture:** Create a diagnostic script that recursively inspects checkpoint contents, identifies non-primitive types, and documents the gap between current serialization and PyTorch 2.9 requirements.

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

Usage:
    python scripts/checkpoint_audit.py <checkpoint.pt>
    python scripts/checkpoint_audit.py --generate-test

Examples:
    # Audit an existing checkpoint
    python scripts/checkpoint_audit.py checkpoints/agent.pt

    # Generate a test checkpoint and audit it
    python scripts/checkpoint_audit.py --generate-test
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# Types that are safe for weights_only=True loading
SAFE_TYPES = (
    int,
    float,
    str,
    bool,
    type(None),
    list,
    tuple,
    dict,
    torch.Tensor,
    # PyTorch internal types that are allowlisted
    torch.dtype,
    torch.device,
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
        issues.append(f"{path}: datetime.datetime → convert to float (timestamp)")
    elif isinstance(value, deque):
        issues.append(f"{path}: collections.deque → convert to list")
        # Still recurse into deque contents
        for i, v in enumerate(value):
            audit_value(v, f"{path}[{i}]", issues, depth + 1, max_depth)
    elif isinstance(value, Enum):
        issues.append(
            f"{path}: Enum {value_type.__module__}.{value_type.__name__}.{value.name} "
            f"→ convert to str (enum name) or int (enum value)"
        )
    elif is_dataclass(value) and not isinstance(value, type):
        issues.append(
            f"{path}: dataclass {value_type.__module__}.{value_type.__name__} "
            f"→ convert to dict"
        )
        # Recurse into dataclass fields
        for field_name in value.__dataclass_fields__:
            field_value = getattr(value, field_name)
            audit_value(field_value, f"{path}.{field_name}", issues, depth + 1, max_depth)
    elif isinstance(value, nn.Module):
        issues.append(
            f"{path}: nn.Module {value_type.__module__}.{value_type.__name__} "
            f"→ store weights in state_dict, config separately"
        )
    elif isinstance(value, dict):
        for k, v in value.items():
            key_repr = repr(k) if not isinstance(k, str) else k
            audit_value(v, f"{path}['{key_repr}']", issues, depth + 1, max_depth)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            audit_value(v, f"{path}[{i}]", issues, depth + 1, max_depth)
    elif not isinstance(value, SAFE_TYPES):
        issues.append(
            f"{path}: UNKNOWN TYPE {value_type.__module__}.{value_type.__name__} "
            f"→ needs manual inspection"
        )

    return issues


def print_section(title: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{title}")
    print(char * len(title))


def audit_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Audit a checkpoint file and return results.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict with audit results including issues, top_level_keys, and error info
    """
    results: dict[str, Any] = {
        "path": str(checkpoint_path),
        "weights_only_compatible": False,
        "weights_only_error": None,
        "issues": [],
        "top_level_keys": {},
    }

    print(f"Auditing: {checkpoint_path}")
    print("=" * 60)

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
            print(f"  • {issue}")
    else:
        print("No issues found (but weights_only=True still failed?)")
        print("This may indicate a PyTorch version mismatch or custom unpickler issue.")

    # Step 4: Show top-level structure
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
            print(f"  • {key}: {type_name}")
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

    This exercises the full extra_state serialization path.
    """
    print("Generating MorphogeneticModel checkpoint...")
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
        # Generate and audit MorphogeneticModel checkpoint
        path = generate_morphogenetic_checkpoint()
        print("\n")
        audit_checkpoint(path)
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
- Generating test checkpoints for analysis
- Detailed reporting of problematic types with conversion hints"
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

1. Testing weights_only=True
=============================
FAILED: ...

2. Loading with weights_only=False
===================================
...
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

## Task 4: Generate and audit MorphogeneticModel checkpoint

**Files:**
- None (diagnostic task)

**Step 1: Generate MorphogeneticModel checkpoint**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-morphogenetic
```

Expected: This exercises `SeedSlot.get_extra_state()` which stores:
- `SeedState` dataclass
- `alpha_schedule` (BlendAlgorithm nn.Module)
- `isolate_gradients` (bool)

**Step 2: Capture output to file**

Run:
```bash
PYTHONPATH=src uv run python scripts/checkpoint_audit.py --generate-morphogenetic 2>&1 | tee /tmp/morphogenetic_audit_output.txt
```

**Step 3: Review findings**

Run: `cat /tmp/morphogenetic_audit_output.txt`

This checkpoint should reveal the core issues in `SeedSlot.get_extra_state()`.

---

## Task 5: Document audit findings

**Files:**
- Create: `docs/plans/kasmina-coordinate-migration/checkpoint-audit-results.md`

**Step 1: Create findings document**

Based on the audit outputs from Tasks 3 and 4, create the results document:

```markdown
# Checkpoint Audit Results

**Date:** 2025-12-16
**Auditor:** [Your name]
**PyTorch Version:** [output of `python -c "import torch; print(torch.__version__)"`]

---

## Summary

[Fill in after running audits]

| Checkpoint Type | weights_only=True | Issues Found |
|-----------------|-------------------|--------------|
| PPO Agent       | PASS/FAIL         | N issues     |
| MorphogeneticModel | PASS/FAIL      | N issues     |

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

## MorphogeneticModel Checkpoint

**Generated:** `/tmp/esper_morphogenetic_checkpoint.pt`

### weights_only=True Result

```
[Paste exact error message]
```

### Issues Found

[List each issue from audit output, especially SeedSlot.get_extra_state issues]

### Top-Level Structure

[Paste top-level keys section]

---

## Type Conversion Requirements

Based on the audit, these types need conversion for M4:

| Current Type | Location | Convert To |
|--------------|----------|------------|
| [Type]       | [Path]   | [Target]   |

---

## Recommended Changes for M4

### SeedSlot.get_extra_state()

Current:
```python
return {
    "seed_state": self.state,  # SeedState dataclass
    "alpha_schedule": self.alpha_schedule,  # BlendAlgorithm nn.Module
    "isolate_gradients": self.isolate_gradients,  # bool (OK)
}
```

Required:
```python
return {
    "seed_state": self.state.to_dict(),  # Pure dict
    "alpha_schedule_config": self._serialize_schedule(),  # Config dict only
    "isolate_gradients": self.isolate_gradients,
}
```

### SeedState

Add `to_dict()` and `from_dict()` methods that convert:
- `stage: SeedStage` → `stage: str` (enum name)
- `stage_entered_at: datetime` → `stage_entered_at: float` (timestamp)
- `stage_history: deque` → `stage_history: list`

---

## Verification

After M4 implementation, re-run audits to confirm:
```bash
PYTHONPATH=src python scripts/checkpoint_audit.py --generate-morphogenetic
```

Expected: `SUCCESS - checkpoint is already compatible!`
```

**Step 2: Commit findings document**

```bash
git add docs/plans/kasmina-coordinate-migration/checkpoint-audit-results.md
git commit -m "docs: add checkpoint audit results for M4 planning

Documents PyTorch 2.9 weights_only=True compatibility issues found
in PPO and MorphogeneticModel checkpoints. Provides conversion
requirements for SeedSlot.get_extra_state()."
```

---

## Task 6: Final verification

**Step 1: Verify all outputs exist**

Run:
```bash
ls -la scripts/checkpoint_audit.py
ls -la docs/plans/kasmina-coordinate-migration/checkpoint-audit-results.md
```

Expected: Both files exist

**Step 2: Run audit tool help**

Run: `PYTHONPATH=src uv run python scripts/checkpoint_audit.py --help`

Expected: Help text displays correctly

**Step 3: Final commit (if any uncommitted changes)**

Run: `git status`

If clean, WP-A is complete.

---

## Acceptance Checklist

- [ ] `scripts/checkpoint_audit.py` exists and runs
- [ ] PPO checkpoint audited, output captured
- [ ] MorphogeneticModel checkpoint audited, output captured
- [ ] `checkpoint-audit-results.md` documents all findings
- [ ] Type conversion requirements clearly specified
- [ ] All changes committed

---

## Outputs

1. **`scripts/checkpoint_audit.py`** — Reusable diagnostic tool
2. **`docs/plans/kasmina-coordinate-migration/checkpoint-audit-results.md`** — Findings document

These outputs directly inform M4 implementation scope.
